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
//! adds the file index offset of the bundle to the file attributes. If a file
//! index does not fit in the form of its attribute, the attribute gets a
//! larger form. If a value changes size, the merge changes the unit length
//! and the references to the entries after the value.
//!
//! The merge supports the DWARF that LLVM writes for PTX: 32-bit DWARF 2 to
//! 4 that refers to labels, and to `.debug_abbrev` and `.debug_line` at
//! offset 0. In DWARF 3 and 4, location expressions can refer to entries.
//! Thus the merge changes the size of values only in DWARF 2. The merged
//! module does not have the name tables `.debug_pubnames` and
//! `.debug_pubtypes`, which LLVM 21 writes. If the merge does not support the
//! DWARF of the first bundle with DWARF sections, only that bundle keeps
//! them, as they are. Another bundle with DWARF that the merge does not
//! support keeps only its line information.

use std::collections::{HashMap, HashSet};
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

/// The data forms and their sizes, from the smallest.
const DATA_FORMS: [(usize, u64); 4] = [
    (1, DW_FORM_DATA1),
    (2, DW_FORM_DATA2),
    (4, DW_FORM_DATA4),
    (8, DW_FORM_DATA8),
];

/// Section ranges, in order, and their new bytes.
type Patches = Vec<(Range<usize>, Vec<u8>)>;

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
        match Units::read(document, &sections) {
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

    /// The new text of the `.debug_info` values of the bundle at `index`: the
    /// codes of the merged table, the file indices plus `file_index_offset`,
    /// and the offsets that change. The merged table gets the declarations
    /// that the bundle adds. `None` if the bundle does not keep its DWARF
    /// sections.
    pub(super) fn replacements(
        &mut self,
        index: usize,
        file_index_offset: u64,
    ) -> Option<Vec<(Range<usize>, String)>> {
        let units = self.units.get(&index)?;
        let info = &units.info;
        // The new index of a file attribute. File index 0 is no file.
        let file_index = |field: &Range<usize>, form: u64| {
            let index = info.number(field, form == DW_FORM_UDATA)?;
            if index == 0 {
                Some(0)
            } else {
                index.checked_add(file_index_offset)
            }
        };

        // A file attribute gets the smallest data form that has the indices
        // of all its entries, as LLVM selects the form.
        let mut forms: Vec<Vec<u64>> = units
            .declarations
            .iter()
            .map(|declaration| {
                declaration
                    .attributes
                    .iter()
                    .map(|&(_, form)| form)
                    .collect()
            })
            .collect();
        for (field, kind) in units.units.iter().flat_map(|unit| &unit.fields) {
            if let Field::File(declaration, attribute) = *kind {
                let form = &mut forms[declaration][attribute];
                let index = file_index(field, *form)?;
                if let Some(current) = DATA_FORMS.iter().position(|&(_, data)| data == *form) {
                    let fits = DATA_FORMS
                        .iter()
                        .position(|&(size, _)| size == 8 || index >> (8 * size) == 0)?;
                    *form = DATA_FORMS[current.max(fits)].1;
                }
            }
        }

        // Each declaration that the merged table does not have gets the next
        // code.
        let mut added: Vec<Vec<u8>> = Vec::new();
        let codes: Vec<u64> = units
            .declarations
            .iter()
            .zip(&forms)
            .map(|(declaration, forms)| {
                let bytes = declaration.bytes(forms);
                if let Some(&code) = self.codes.get(&bytes) {
                    return code;
                }
                let position = match added.iter().position(|added| *added == bytes) {
                    Some(position) => position,
                    None => {
                        added.push(bytes);
                        added.len() - 1
                    }
                };
                (self.declarations.len() + position + 1) as u64
            })
            .collect();

        let mut patches = Vec::new();
        for unit in &units.units {
            patches.extend(unit.patches(info, &codes, &forms, file_index)?);
        }
        let replacements = info.rewrite(&patches)?;
        for bytes in added {
            self.declarations.push(bytes.clone());
            self.codes.insert(bytes, self.declarations.len() as u64);
        }
        Some(replacements)
    }

    /// The names that the compile units of the bundle at `index` refer to
    /// and that the bundle defines in a callable. The merge renames them in
    /// each bundle.
    pub(super) fn names(&self, index: usize) -> &[&'source str] {
        self.units
            .get(&index)
            .map_or(&[], |units| units.names.as_slice())
    }

    /// Whether a bundle that keeps its DWARF sections keeps the section
    /// `name`. The name tables refer to the offsets of the compile units,
    /// which the merge changes.
    pub(super) fn keeps(&self, name: &str) -> bool {
        self.unchanged || !matches!(name, ".debug_pubnames" | ".debug_pubtypes")
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
    /// The declarations of the abbreviation table of the bundle.
    declarations: Vec<Declaration>,
    units: Vec<Unit>,
    /// The names that the units refer to and that the bundle defines in a
    /// callable.
    names: Vec<&'source str>,
}

impl<'source> Units<'source> {
    /// The compile units in the DWARF `sections` of `document`. `None` if the
    /// merge does not support them.
    fn read(document: &Document<'source>, sections: &[Section<'source>]) -> Option<Self> {
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
        // The index of the declaration of each code.
        let mut codes = HashMap::new();
        for (index, (code, _)) in declarations.iter().enumerate() {
            if codes.insert(*code, index).is_some() {
                return None;
            }
        }
        let declarations: Vec<_> = declarations
            .into_iter()
            .map(|(_, declaration)| declaration)
            .collect();
        let units = walk(&info, &codes, &declarations)?;

        let private = private_names(document);
        let names = info
            .names
            .iter()
            .copied()
            .filter(|name| private.contains(name))
            .collect();
        Some(Self {
            info,
            declarations,
            units,
            names,
        })
    }

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
            declarations: Vec::new(),
            units: Vec::new(),
            names,
        }
    }
}

/// A compile unit.
struct Unit {
    /// The section range of the unit.
    range: Range<usize>,
    version: u64,
    /// The values that the merge can change, in order.
    fields: Vec<(Range<usize>, Field)>,
}

/// A value in a compile unit that the merge can change.
enum Field {
    /// The abbreviation code of an entry, with the index of its declaration.
    Code(usize),
    /// A file attribute, with the indices of its declaration and attribute.
    File(usize, usize),
    /// A reference to an entry, with its form.
    Reference(u64),
}

impl Unit {
    /// The new bytes of the values of the unit: the `codes` of the
    /// declarations, and the file indices from `file_index` in the `forms`
    /// of the declarations. If a value changes size, also the unit length
    /// and the references. `None` if a value does not fit.
    fn patches(
        &self,
        info: &SectionData,
        codes: &[u64],
        forms: &[Vec<u64>],
        file_index: impl Fn(&Range<usize>, u64) -> Option<u64>,
    ) -> Option<Patches> {
        let mut patches = Vec::new();
        for (field, kind) in &self.fields {
            let bytes = match *kind {
                Field::Code(declaration) => uleb128(codes[declaration]),
                Field::File(declaration, attribute) => {
                    let form = forms[declaration][attribute];
                    let index = file_index(field, form)?;
                    match DATA_FORMS.iter().find(|&&(_, data)| data == form) {
                        Some(&(size, _)) => le_bytes(index, size)?,
                        None => uleb128(index),
                    }
                }
                Field::Reference(_) => continue,
            };
            if !info.bytes[field.clone()]
                .iter()
                .copied()
                .eq(bytes.iter().copied().map(Some))
            {
                patches.push((field.clone(), bytes));
            }
        }
        if patches
            .iter()
            .all(|(field, bytes)| field.len() == bytes.len())
        {
            return Some(patches);
        }
        // In DWARF 3 and 4, location expressions can refer to entries.
        if self.version != 2 {
            return None;
        }

        // The change of the offsets after each new value.
        let mut change = 0;
        let changes: Vec<(usize, i64)> = patches
            .iter()
            .map(|(field, bytes)| {
                change += bytes.len() as i64 - field.len() as i64;
                (field.end, change)
            })
            .collect();
        // The new unit offset of a unit offset.
        let moved = |offset: u64| {
            let position = self
                .range
                .start
                .checked_add(usize::try_from(offset).ok()?)?;
            let count = changes.partition_point(|&(end, _)| end <= position);
            offset.checked_add_signed(count.checked_sub(1).map_or(0, |last| changes[last].1))
        };
        for (field, kind) in &self.fields {
            let Field::Reference(form) = *kind else {
                continue;
            };
            let udata = form == DW_FORM_REF_UDATA;
            let offset = info.number(field, udata)?;
            let new_offset = moved(offset)?;
            // The reference keeps its size, thus the offsets do not change
            // again.
            let bytes = if udata {
                Some(uleb128(new_offset)).filter(|bytes| bytes.len() == field.len())
            } else {
                le_bytes(new_offset, field.len())
            }?;
            if new_offset != offset {
                patches.push((field.clone(), bytes));
            }
        }
        // 32-bit DWARF only.
        let length = moved(self.range.len() as u64)?
            .checked_sub(4)
            .filter(|&length| length < 0xffff_fff0)?;
        patches.push((self.range.start..self.range.start + 4, le_bytes(length, 4)?));
        patches.sort_by_key(|(field, _)| field.start);
        Some(patches)
    }
}

/// The values of the data directives in the DWARF sections with one name.
struct SectionData<'source> {
    source: &'source str,
    /// The data directives.
    directives: Vec<Data>,
    /// The source spans of the values.
    spans: Vec<Range<usize>>,
    /// The section offset of each value.
    offsets: Vec<usize>,
    /// The section bytes. `None` is a byte of a value that is not a number.
    bytes: Vec<Option<u8>>,
    /// The names that the values refer to.
    names: Vec<&'source str>,
}

/// A data directive.
struct Data {
    /// The source span of the directive.
    span: Range<usize>,
    /// The source span of the values.
    arguments: Range<usize>,
    /// The size of each value.
    size: usize,
    /// The section range of the values.
    bytes: Range<usize>,
}

impl<'source> SectionData<'source> {
    fn new(source: &'source str) -> Self {
        Self {
            source,
            directives: Vec::new(),
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
            let start = self.bytes.len();
            for value in tokens.split(|token| token.text(self.source) == ",") {
                self.push(value, size)?;
            }
            self.directives.push(Data {
                span: directive.span(),
                arguments: directive.arguments_span(),
                size,
                bytes: start..self.bytes.len(),
            });
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

    /// The number in the section range `field`: a ULEB128 number, or a
    /// little-endian number of the size of the range.
    fn number(&self, field: &Range<usize>, uleb: bool) -> Option<u64> {
        let mut reader = Reader {
            bytes: &self.bytes[..field.end],
            position: field.start,
        };
        if uleb {
            reader.uleb128()
        } else {
            reader.number(field.len())
        }
    }

    /// The new text of the data directives with the bytes of `patches`. A
    /// directive gets the new bytes of a range that starts in it, and loses
    /// the other bytes of the range. `None` if a directive cannot have its
    /// new bytes.
    fn rewrite(&self, patches: &Patches) -> Option<Vec<(Range<usize>, String)>> {
        let mut replacements = Vec::new();
        let mut next = 0;
        for data in &self.directives {
            let mut bytes = Vec::new();
            let mut changed = false;
            let mut position = data.bytes.start;
            while position < data.bytes.end {
                match patches.get(next) {
                    Some((field, new)) if field.start <= position => {
                        if field.start >= data.bytes.start {
                            bytes.extend(new.iter().copied().map(Some));
                        }
                        changed = true;
                        position = field.end;
                        if field.end <= data.bytes.end {
                            next += 1;
                        }
                    }
                    _ => {
                        bytes.push(self.bytes[position]);
                        position += 1;
                    }
                }
            }
            if !changed {
                continue;
            }
            // Only the values that are numbers change.
            let bytes: Vec<u8> = bytes.into_iter().collect::<Option<_>>()?;
            if !bytes.len().is_multiple_of(data.size) {
                return None;
            }
            let values: Vec<String> = bytes
                .chunks(data.size)
                .map(|value| {
                    let number = value
                        .iter()
                        .rev()
                        .fold(0, |number, &byte| number << 8 | u64::from(byte));
                    number.to_string()
                })
                .collect();
            replacements.push(if values.is_empty() {
                (data.span.clone(), String::new())
            } else {
                (data.arguments.clone(), values.join(", "))
            });
        }
        Some(replacements)
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
    tag: u64,
    children: u8,
    /// The attributes and their forms.
    attributes: Vec<(u64, u64)>,
}

impl Declaration {
    /// The bytes after the code, with `forms` as the forms of the
    /// attributes.
    fn bytes(&self, forms: &[u64]) -> Vec<u8> {
        let mut bytes = uleb128(self.tag);
        bytes.push(self.children);
        for (&(attribute, _), &form) in self.attributes.iter().zip(forms) {
            bytes.extend(uleb128(attribute));
            bytes.extend(uleb128(form));
        }
        bytes.extend([0, 0]);
        bytes
    }
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
        let tag = reader.uleb128()?;
        let children = reader.byte()?;
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
        declarations.push((
            code,
            Declaration {
                tag,
                children,
                attributes,
            },
        ));
    }
    (reader.position == abbrev.bytes.len()).then_some(declarations)
}

/// The compile units of `info`. `codes` has the index in `declarations` of
/// each abbreviation code. `None` if the merge does not support the units.
fn walk(
    info: &SectionData,
    codes: &HashMap<u64, usize>,
    declarations: &[Declaration],
) -> Option<Vec<Unit>> {
    let mut units = Vec::new();
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
        let mut fields = Vec::new();
        while unit.position < end {
            let start = unit.position;
            let code = unit.uleb128()?;
            // The end of a list of children.
            if code == 0 {
                continue;
            }
            let &declaration = codes.get(&code)?;
            fields.push((start..unit.position, Field::Code(declaration)));
            for (index, &(attribute, form)) in
                declarations[declaration].attributes.iter().enumerate()
            {
                let value = unit.position;
                unit.skip_value(form, version, address_size)?;
                let field = value..unit.position;
                let number = info.bytes[field.clone()].iter().all(Option::is_some);
                match (attribute, form) {
                    (
                        DW_AT_DECL_FILE | DW_AT_CALL_FILE,
                        DW_FORM_DATA1 | DW_FORM_DATA2 | DW_FORM_DATA4 | DW_FORM_DATA8
                        | DW_FORM_UDATA,
                    ) if number => fields.push((field, Field::File(declaration, index))),
                    (DW_AT_DECL_FILE | DW_AT_CALL_FILE, _) => return None,
                    (
                        _,
                        DW_FORM_REF1 | DW_FORM_REF2 | DW_FORM_REF4 | DW_FORM_REF8
                        | DW_FORM_REF_UDATA,
                    ) => fields.push((field, Field::Reference(form))),
                    // The merge puts the sections of earlier bundles before
                    // this offset.
                    (_, DW_FORM_STRP | DW_FORM_REF_ADDR | DW_FORM_SEC_OFFSET) if number => {
                        return None;
                    }
                    _ => {}
                }
            }
        }
        units.push(Unit {
            range: offset..end,
            version,
            fields,
        });
        offset = end;
    }
    Some(units)
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

/// `number` as a little-endian number of `size` bytes. `None` if it does not
/// fit.
fn le_bytes(number: u64, size: usize) -> Option<Vec<u8>> {
    (size == 8 || number >> (8 * size) == 0).then(|| number.to_le_bytes()[..size].to_vec())
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
