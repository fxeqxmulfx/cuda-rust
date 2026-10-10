/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Merge the DWARF sections of PTX bundles.
//!
//! ptxas reads all compile units with the abbreviation table at the start of
//! `.debug_abbrev`. It makes one line program from the `.file` and `.loc`
//! directives of the module. Thus the merged module has one abbreviation
//! table with the declarations of all bundles, and one `.debug_info` section
//! with the compile units of all bundles. In each compile unit, the merge
//! changes the abbreviation codes to the codes of the merged table. It also
//! adds the file index offset of the bundle to the file attributes.
//!
//! The merge supports the DWARF that LLVM writes for PTX: 32-bit DWARF 2 to
//! 4 that refers to labels, and to `.debug_abbrev` and `.debug_line` at
//! offset 0. The merged module does not have the name tables
//! `.debug_pubnames` and `.debug_pubtypes`, which LLVM 21 writes. If the
//! merge does not support the DWARF of the first bundle with DWARF sections,
//! only that bundle keeps them, as they are. Another bundle with DWARF that
//! the merge does not support keeps only its line information.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::ops::Range;

use ptx_parse::{Document, ScopeId, StatementKind, Token, TokenKind};

use super::{declared_variables, span_tokens};

const DW_AT_DECL_FILE: u64 = 0x3a;
const DW_AT_CALL_FILE: u64 = 0x58;

const DW_FORM_ADDR: u64 = 0x01;
const DW_FORM_BLOCK2: u64 = 0x03;
const DW_FORM_BLOCK4: u64 = 0x04;
const DW_FORM_DATA2: u64 = 0x05;
const DW_FORM_DATA4: u64 = 0x06;
const DW_FORM_DATA8: u64 = 0x07;
const DW_FORM_STRING: u64 = 0x08;
const DW_FORM_BLOCK: u64 = 0x09;
const DW_FORM_BLOCK1: u64 = 0x0a;
const DW_FORM_DATA1: u64 = 0x0b;
const DW_FORM_FLAG: u64 = 0x0c;
const DW_FORM_SDATA: u64 = 0x0d;
const DW_FORM_STRP: u64 = 0x0e;
const DW_FORM_UDATA: u64 = 0x0f;
const DW_FORM_REF_ADDR: u64 = 0x10;
const DW_FORM_REF1: u64 = 0x11;
const DW_FORM_REF2: u64 = 0x12;
const DW_FORM_REF4: u64 = 0x13;
const DW_FORM_REF8: u64 = 0x14;
const DW_FORM_REF_UDATA: u64 = 0x15;
const DW_FORM_SEC_OFFSET: u64 = 0x17;
const DW_FORM_EXPRLOC: u64 = 0x18;
const DW_FORM_FLAG_PRESENT: u64 = 0x19;
const DW_FORM_REF_SIG8: u64 = 0x20;

/// Values in a section, by section range, each with a number: a new code or a
/// form.
type Fields = Vec<(Range<usize>, u64)>;

/// The new code and the attributes and forms of each abbreviation code of a
/// bundle.
type Codes<'declarations> = HashMap<u64, (u64, &'declarations [(u64, u64)])>;

/// A DWARF section of a bundle.
pub(super) struct Section<'source> {
    pub(super) name: &'source str,
    /// The `.section` line through the closing brace and its newline.
    pub(super) span: Range<usize>,
    /// The text between the braces.
    pub(super) body: Range<usize>,
    scope: Option<ScopeId>,
}

/// The DWARF sections of `document`, in source order.
pub(super) fn sections<'source>(document: &Document<'source>) -> Vec<Section<'source>> {
    let ptx = document.source();
    document
        .directives()
        .iter()
        .filter(|directive| directive.name() == ".section")
        .filter_map(|directive| {
            let name = directive
                .arguments()
                .split(|c: char| c.is_whitespace() || c == '{')
                .find(|name| !name.is_empty())
                .filter(|name| name.starts_with(".debug_"))?;
            let line = directive.line_span();
            let scope = document
                .scopes()
                .iter()
                .find(|scope| scope.header() == Some(directive.statement()));
            let (end, body) = scope.map_or((line.end, line.end..line.end), |scope| {
                let end = scope.span().end;
                (
                    line.end
                        .max(end + usize::from(ptx[end..].starts_with('\n'))),
                    scope.body_span(),
                )
            });
            Some(Section {
                name,
                span: line.start..end,
                body,
                scope: scope.map(|scope| scope.id()),
            })
        })
        .collect()
}

/// The merged abbreviation table, and the compile units of the bundles that
/// keep their DWARF sections.
#[derive(Default)]
pub(super) struct DwarfMerge<'source> {
    /// The declarations of the merged table, without their codes. The code
    /// of a declaration is its index plus one.
    declarations: Vec<Vec<u8>>,
    /// The code of each declaration in `declarations`.
    codes: HashMap<Vec<u8>, u64>,
    /// The compile units of each bundle that keeps its DWARF sections, by
    /// bundle index.
    units: HashMap<usize, Units<'source>>,
    /// Whether the first bundle with DWARF sections keeps them as they are,
    /// because the merge does not support them.
    unchanged: bool,
}

impl<'source> DwarfMerge<'source> {
    /// Read the DWARF sections of the bundle at `index`. The bundle keeps
    /// them if the merge supports them. If the merge does not support the
    /// sections of the first bundle with DWARF sections, only that bundle
    /// keeps them, as they are.
    pub(super) fn add(&mut self, index: usize, document: &Document<'source>) {
        let sections = sections(document);
        if sections.is_empty() || self.unchanged {
            return;
        }
        match self.read(document, &sections) {
            Some(units) => {
                self.units.insert(index, units);
            }
            None if self.units.is_empty() => {
                self.unchanged = true;
                self.units
                    .insert(index, Units::unchanged(document, &sections));
            }
            None => {}
        }
    }

    /// The compile units of the bundle at `index`, if it keeps its DWARF
    /// sections.
    pub(super) fn units(&self, index: usize) -> Option<&Units<'source>> {
        self.units.get(&index)
    }

    /// Whether a bundle that keeps its DWARF sections keeps the section
    /// `name`. The name tables refer to the offsets of the compile units,
    /// which the merge changes.
    pub(super) fn keeps(&self, name: &str) -> bool {
        self.unchanged || !matches!(name, ".debug_pubnames" | ".debug_pubtypes")
    }

    fn read(
        &mut self,
        document: &Document<'source>,
        sections: &[Section<'source>],
    ) -> Option<Units<'source>> {
        let mut abbrev = SectionData::new(document.source());
        let mut info = SectionData::new(document.source());
        for section in sections {
            let scope = section.scope?;
            match section.name {
                ".debug_abbrev" => abbrev.read(document, scope)?,
                ".debug_info" => info.read(document, scope)?,
                // `.loc` directives refer to the strings with labels.
                ".debug_str" => {}
                // The merged module does not have them.
                ".debug_pubnames" | ".debug_pubtypes" => {}
                // The merge cannot find the references to other sections.
                _ if document.statements_in_scope(scope).next().is_some() => return None,
                _ => {}
            }
        }
        let declarations = if abbrev.bytes.is_empty() {
            Vec::new()
        } else {
            declarations(&abbrev)?
        };

        // Each declaration that the merged table does not have gets the next
        // code.
        let mut added: Vec<&[u8]> = Vec::new();
        let mut codes = HashMap::new();
        for (code, declaration) in &declarations {
            let new_code = match self.codes.get(&declaration.bytes) {
                Some(&new_code) => new_code,
                None => {
                    let position = match added.iter().position(|&bytes| bytes == declaration.bytes)
                    {
                        Some(position) => position,
                        None => {
                            added.push(&declaration.bytes);
                            added.len() - 1
                        }
                    };
                    (self.declarations.len() + position + 1) as u64
                }
            };
            if codes
                .insert(*code, (new_code, declaration.attributes.as_slice()))
                .is_some()
            {
                return None;
            }
        }
        let (changed_codes, files) = walk(&info, &codes)?;
        for bytes in added {
            self.declarations.push(bytes.to_vec());
            self.codes
                .insert(bytes.to_vec(), self.declarations.len() as u64);
        }

        let private = private_names(document);
        let names = info
            .names
            .iter()
            .copied()
            .filter(|name| private.contains(name))
            .collect();
        Some(Units {
            info,
            codes: changed_codes,
            files,
            names,
        })
    }

    /// Write the DWARF sections of the merged module. `sections` has the
    /// contents of the sections that the bundles keep, by name in the order
    /// of the first bundle with the name. `.debug_abbrev` gets the merged
    /// table.
    pub(super) fn write_sections(&self, sections: &[(&str, Vec<String>)], output: &mut String) {
        for (name, contents) in sections {
            let mut body = String::new();
            if *name == ".debug_abbrev" && !self.unchanged {
                for (code, declaration) in (1..).zip(&self.declarations) {
                    for byte in uleb128(code).iter().chain(declaration) {
                        let _ = writeln!(body, ".b8 {byte}");
                    }
                }
                body.push_str(".b8 0\n");
            } else {
                for content in contents.iter().filter(|content| !content.is_empty()) {
                    body.push_str(content);
                    body.push('\n');
                }
            }
            if body.is_empty() {
                let _ = writeln!(output, "\t.section\t{name}\t{{\t}}");
            } else {
                let _ = write!(output, "\t.section\t{name}\n\t{{\n{body}\t}}\n");
            }
        }
    }
}

/// The compile units of a bundle.
pub(super) struct Units<'source> {
    info: SectionData<'source>,
    /// The abbreviation codes that change, and their new codes.
    codes: Fields,
    /// The file attributes and their forms.
    files: Fields,
    /// The names that the units refer to and that the bundle defines in a
    /// callable.
    names: Vec<&'source str>,
}

impl<'source> Units<'source> {
    /// The units of a bundle that keeps its DWARF `sections` as they are.
    fn unchanged(document: &Document<'source>, sections: &[Section<'source>]) -> Self {
        let private = private_names(document);
        let names = sections
            .iter()
            .flat_map(|section| span_tokens(document, section.body.clone()))
            .map(|token| token.text(document.source()))
            .filter(|name| private.contains(name))
            .collect();
        Self {
            info: SectionData::new(document.source()),
            codes: Vec::new(),
            files: Vec::new(),
            names,
        }
    }

    /// The new text of the `.debug_info` values: the codes of the merged
    /// table, and the file indices plus `file_index_offset`. `None` if a file
    /// index does not fit in its value.
    pub(super) fn replacements(
        &self,
        file_index_offset: u64,
    ) -> Option<Vec<(Range<usize>, String)>> {
        let mut numbers = BTreeMap::new();
        for (field, code) in &self.codes {
            self.info.write(&mut numbers, field.start, &uleb128(*code));
        }
        for (field, form) in &self.files {
            let mut reader = Reader {
                bytes: &self.info.bytes[..field.end],
                position: field.start,
            };
            let index = if *form == DW_FORM_UDATA {
                reader.uleb128()?
            } else {
                reader.number(field.len())?
            };
            // File index 0 is no file.
            if index == 0 || file_index_offset == 0 {
                continue;
            }
            let index = index.checked_add(file_index_offset)?;
            // The value keeps its size, thus the offsets of the next values
            // do not change.
            let bytes = if *form == DW_FORM_UDATA {
                Some(uleb128(index)).filter(|bytes| bytes.len() == field.len())
            } else {
                (field.len() == 8 || index >> (8 * field.len()) == 0)
                    .then(|| index.to_le_bytes()[..field.len()].to_vec())
            }?;
            self.info.write(&mut numbers, field.start, &bytes);
        }
        Some(
            numbers
                .into_iter()
                .filter(|&(value, number)| number != self.info.number(value))
                .map(|(value, number)| (self.info.spans[value].clone(), number.to_string()))
                .collect(),
        )
    }

    /// The names that the compile units refer to and that the bundle defines
    /// in a callable. The merge renames them in each bundle.
    pub(super) fn names(&self) -> &[&'source str] {
        &self.names
    }
}

/// The values of the data directives in the DWARF sections with one name.
struct SectionData<'source> {
    source: &'source str,
    /// The source spans of the values.
    spans: Vec<Range<usize>>,
    /// The section offset of each value.
    offsets: Vec<usize>,
    /// The section bytes. `None` is a byte of a value that is not a number.
    bytes: Vec<Option<u8>>,
    /// The names that the values refer to.
    names: Vec<&'source str>,
}

impl<'source> SectionData<'source> {
    fn new(source: &'source str) -> Self {
        Self {
            source,
            spans: Vec::new(),
            offsets: Vec::new(),
            bytes: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Add the values of the data directives in `scope`. `None` if the scope
    /// has other statements.
    fn read(&mut self, document: &Document<'source>, scope: ScopeId) -> Option<()> {
        if document
            .scopes()
            .iter()
            .any(|inner| inner.parent() == Some(scope))
        {
            return None;
        }
        for statement in document.statements_in_scope(scope) {
            if statement.kind() != StatementKind::Directive {
                return None;
            }
            let directive = document.directive_for_statement(statement.id())?;
            let size = match directive.name() {
                ".b8" => 1,
                ".b16" => 2,
                ".b32" => 4,
                ".b64" => 8,
                _ => return None,
            };
            if directive.labels().len() != 0 {
                return None;
            }
            let tokens: Vec<_> = span_tokens(document, directive.arguments_span()).collect();
            for value in tokens.split(|token| token.text(self.source) == ",") {
                self.push(value, size)?;
            }
        }
        Some(())
    }

    /// Add a value of `size` bytes.
    fn push(&mut self, tokens: &[&Token], size: usize) -> Option<()> {
        let span = tokens.first()?.span().start..tokens.last()?.span().end;
        let text = &self.source[span.clone()];
        self.offsets.push(self.bytes.len());
        if text.starts_with(|c: char| c.is_ascii_digit()) {
            let number = match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
                Some(digits) => u64::from_str_radix(digits, 16).ok()?,
                None => text.parse().ok()?,
            };
            if size < 8 && number >> (8 * size) != 0 {
                return None;
            }
            self.bytes
                .extend(number.to_le_bytes()[..size].iter().copied().map(Some));
        } else {
            for token in tokens {
                let word = token.text(self.source);
                if token.kind() != TokenKind::Word || word.starts_with(|c: char| c.is_ascii_digit())
                {
                    continue;
                }
                if !word.starts_with('.') {
                    self.names.push(word);
                } else if tokens.len() != 1 || !matches!(word, ".debug_abbrev" | ".debug_line") {
                    // Only the start of these sections is the same after
                    // the merge.
                    return None;
                }
            }
            self.bytes.extend(std::iter::repeat_n(None, size));
        }
        self.spans.push(span);
        Some(())
    }

    /// The number of the value at `index`. Its bytes are numbers.
    fn number(&self, index: usize) -> u64 {
        let end = self
            .offsets
            .get(index + 1)
            .copied()
            .unwrap_or(self.bytes.len());
        self.bytes[self.offsets[index]..end]
            .iter()
            .rev()
            .fold(0, |number, byte| number << 8 | u64::from(byte.unwrap_or(0)))
    }

    /// Put `bytes` at section `offset` into `numbers`, the new numbers of the
    /// values.
    fn write(&self, numbers: &mut BTreeMap<usize, u64>, offset: usize, bytes: &[u8]) {
        for (position, &byte) in (offset..).zip(bytes) {
            let value = self.offsets.partition_point(|&start| start <= position) - 1;
            let shift = 8 * (position - self.offsets[value]);
            let number = numbers.entry(value).or_insert_with(|| self.number(value));
            *number = *number & !(0xff_u64 << shift) | u64::from(byte) << shift;
        }
    }

    /// Whether the 4 bytes at `offset` are the offset of the first
    /// abbreviation table: `.debug_abbrev` or 0.
    fn is_abbrev_start(&self, offset: usize) -> bool {
        let Some(bytes) = self.bytes.get(offset..offset + 4) else {
            return false;
        };
        if bytes.iter().all(|&byte| byte == Some(0)) {
            return true;
        }
        let value = self.offsets.partition_point(|&start| start <= offset) - 1;
        self.offsets[value] == offset
            && self
                .offsets
                .get(value + 1)
                .copied()
                .unwrap_or(self.bytes.len())
                == offset + 4
            && &self.source[self.spans[value].clone()] == ".debug_abbrev"
    }
}

/// An abbreviation declaration.
struct Declaration {
    /// The bytes after the code: the tag, the children flag and the
    /// attribute specifications.
    bytes: Vec<u8>,
    /// The attributes and their forms.
    attributes: Vec<(u64, u64)>,
}

/// The declarations of the abbreviation table in `abbrev`, with their codes.
/// `None` if the section has more than one table.
fn declarations(abbrev: &SectionData) -> Option<Vec<(u64, Declaration)>> {
    let mut reader = Reader {
        bytes: &abbrev.bytes,
        position: 0,
    };
    let mut declarations = Vec::new();
    loop {
        let code = reader.uleb128()?;
        if code == 0 {
            break;
        }
        let start = reader.position;
        // The tag and the children flag.
        reader.uleb128()?;
        reader.byte()?;
        let mut attributes = Vec::new();
        loop {
            let attribute = reader.uleb128()?;
            let form = reader.uleb128()?;
            if (attribute, form) == (0, 0) {
                break;
            }
            // Only forms that the merge can read. A DW_FORM_implicit_const
            // declaration also has a value.
            if !is_supported_form(form) {
                return None;
            }
            attributes.push((attribute, form));
        }
        let bytes = abbrev.bytes[start..reader.position]
            .iter()
            .flatten()
            .copied()
            .collect();
        declarations.push((code, Declaration { bytes, attributes }));
    }
    (reader.position == abbrev.bytes.len()).then_some(declarations)
}

/// The abbreviation codes that change and the file attributes in the
/// compile units of `info`. `None` if the merge does not support the units.
fn walk(info: &SectionData, codes: &Codes) -> Option<(Fields, Fields)> {
    let mut changed_codes = Vec::new();
    let mut files = Vec::new();
    let mut offset = 0;
    while offset < info.bytes.len() {
        let mut header = Reader {
            bytes: &info.bytes,
            position: offset,
        };
        let length = header.number(4)?;
        // 32-bit DWARF only.
        if length >= 0xffff_fff0 {
            return None;
        }
        let end = header.position.checked_add(usize::try_from(length).ok()?)?;
        if end > info.bytes.len() {
            return None;
        }
        let version = header.number(2)?;
        if !(2..=4).contains(&version) || !info.is_abbrev_start(header.position) {
            return None;
        }
        header.skip(4)?;
        let address_size = header.byte()?;
        let mut unit = Reader {
            bytes: &info.bytes[..end],
            position: header.position,
        };
        while unit.position < end {
            let start = unit.position;
            let code = unit.uleb128()?;
            // The end of a list of children.
            if code == 0 {
                continue;
            }
            let &(new_code, attributes) = codes.get(&code)?;
            // The offsets of the next values must not change.
            if uleb128(new_code).len() != unit.position - start {
                return None;
            }
            if new_code != code {
                changed_codes.push((start..unit.position, new_code));
            }
            for &(attribute, form) in attributes {
                let field = unit.position;
                unit.skip_value(form, version, address_size)?;
                let number = info.bytes[field..unit.position].iter().all(Option::is_some);
                match (attribute, form) {
                    (
                        DW_AT_DECL_FILE | DW_AT_CALL_FILE,
                        DW_FORM_DATA1 | DW_FORM_DATA2 | DW_FORM_DATA4 | DW_FORM_DATA8
                        | DW_FORM_UDATA,
                    ) if number => files.push((field..unit.position, form)),
                    (DW_AT_DECL_FILE | DW_AT_CALL_FILE, _) => return None,
                    // The merge puts the sections of earlier bundles before
                    // this offset.
                    (_, DW_FORM_STRP | DW_FORM_REF_ADDR | DW_FORM_SEC_OFFSET) if number => {
                        return None;
                    }
                    _ => {}
                }
            }
        }
        offset = end;
    }
    Some((changed_codes, files))
}

/// Whether [`Reader::skip_value`] can read values of `form`.
fn is_supported_form(form: u64) -> bool {
    matches!(
        form,
        DW_FORM_ADDR
            | DW_FORM_BLOCK2
            | DW_FORM_BLOCK4
            | DW_FORM_DATA2
            | DW_FORM_DATA4
            | DW_FORM_DATA8
            | DW_FORM_STRING
            | DW_FORM_BLOCK
            | DW_FORM_BLOCK1
            | DW_FORM_DATA1
            | DW_FORM_FLAG
            | DW_FORM_SDATA
            | DW_FORM_STRP
            | DW_FORM_UDATA
            | DW_FORM_REF_ADDR
            | DW_FORM_REF1
            | DW_FORM_REF2
            | DW_FORM_REF4
            | DW_FORM_REF8
            | DW_FORM_REF_UDATA
            | DW_FORM_SEC_OFFSET
            | DW_FORM_EXPRLOC
            | DW_FORM_FLAG_PRESENT
            | DW_FORM_REF_SIG8
    )
}

/// Reads the bytes of a DWARF section.
struct Reader<'data> {
    bytes: &'data [Option<u8>],
    position: usize,
}

impl Reader<'_> {
    /// The next byte. `None` if it is not a number.
    fn byte(&mut self) -> Option<u8> {
        let byte = (*self.bytes.get(self.position)?)?;
        self.position += 1;
        Some(byte)
    }

    /// A little-endian number of `size` bytes.
    fn number(&mut self, size: usize) -> Option<u64> {
        (0..size).try_fold(0, |number, index| {
            Some(number | u64::from(self.byte()?) << (8 * index))
        })
    }

    fn uleb128(&mut self) -> Option<u64> {
        let mut number = 0;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            number |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Some(number);
            }
        }
        None
    }

    fn skip(&mut self, size: u64) -> Option<()> {
        let end = self.position.checked_add(usize::try_from(size).ok()?)?;
        if end > self.bytes.len() {
            return None;
        }
        self.position = end;
        Some(())
    }

    /// Skip a value of `form`.
    fn skip_value(&mut self, form: u64, version: u64, address_size: u8) -> Option<()> {
        let size = match form {
            DW_FORM_ADDR => u64::from(address_size),
            DW_FORM_REF_ADDR if version == 2 => u64::from(address_size),
            DW_FORM_DATA1 | DW_FORM_FLAG | DW_FORM_REF1 => 1,
            DW_FORM_DATA2 | DW_FORM_REF2 => 2,
            DW_FORM_DATA4 | DW_FORM_REF4 | DW_FORM_STRP | DW_FORM_REF_ADDR | DW_FORM_SEC_OFFSET => {
                4
            }
            DW_FORM_DATA8 | DW_FORM_REF8 | DW_FORM_REF_SIG8 => 8,
            DW_FORM_FLAG_PRESENT => 0,
            DW_FORM_BLOCK1 => u64::from(self.byte()?),
            DW_FORM_BLOCK2 => self.number(2)?,
            DW_FORM_BLOCK4 => self.number(4)?,
            DW_FORM_BLOCK | DW_FORM_EXPRLOC => self.uleb128()?,
            DW_FORM_UDATA | DW_FORM_REF_UDATA => {
                self.uleb128()?;
                0
            }
            DW_FORM_SDATA => {
                while self.byte()? & 0x80 != 0 {}
                0
            }
            DW_FORM_STRING => {
                while self.byte()? != 0 {}
                0
            }
            _ => return None,
        };
        self.skip(size)
    }
}

/// The ULEB128 encoding of `number`.
fn uleb128(mut number: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let byte = (number & 0x7f) as u8;
        number >>= 7;
        if number == 0 {
            bytes.push(byte);
            return bytes;
        }
        bytes.push(byte | 0x80);
    }
}

/// The names that `document` defines in a callable or a section: labels and
/// variables.
fn private_names<'source>(document: &Document<'source>) -> HashSet<&'source str> {
    let ptx = document.source();
    let labels = document
        .labels()
        .iter()
        .filter(|label| label.scope() != ScopeId::ROOT)
        .map(|label| label.name());
    let variables = document
        .directives()
        .iter()
        .filter(|directive| {
            directive.scope() != ScopeId::ROOT
                && matches!(
                    directive.name(),
                    ".local" | ".shared" | ".global" | ".const"
                )
        })
        .flat_map(|directive| {
            declared_variables(ptx, span_tokens(document, directive.arguments_span()))
        });
    labels.chain(variables).collect()
}
