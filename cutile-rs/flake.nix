{
  description = "Development shell for cutile-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        isDarwin = pkgs.stdenv.hostPlatform.isDarwin;

        # CUDA 13.4.2 fetched from nvidia (headers needed on all platforms for
        # bindgen type generation; runtime libraries only needed on Linux)
        cudaRedistBase = "https://developer.download.nvidia.com/compute/cuda/redist";
        # NVIDIA publishes per-architecture redist archives; aarch64 (server) uses
        # the "linux-sbsa" variant, everything else uses "linux-x86_64".
        cudaRedistArch = if pkgs.stdenv.hostPlatform.isAarch64 then "linux-sbsa" else "linux-x86_64";
        fetchCudaRedist =
          { name, version, sha256 }:
          pkgs.fetchurl {
            url = "${cudaRedistBase}/${name}/${cudaRedistArch}/${name}-${cudaRedistArch}-${version}-archive.tar.xz";
            sha256 = sha256.${cudaRedistArch};
          };
        # Hashes come from NVIDIA's manifest for this release:
        # https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.2.json
        # (per package: .<name>.<arch>.sha256). Update from there on version bumps.
        #
        # ORDER MATTERS: archives are extracted with --skip-old-files
        # (first-wins), so components must come before other archives that
        # may carry stale copies of their files — libnvvm before cuda_nvcc,
        # since both can populate nvvm/.
        cudaRedistArchives = [
          (fetchCudaRedist {
            name = "cuda_crt";
            version = "13.4.92";
            sha256 = {
              linux-x86_64 = "c969e61ded12dbf0cf20ace72867f1622ec0e8ea1e1d305e27761012025f7aaa";
              linux-sbsa = "0358db52257885611664240e030e45528f052aa1360c886f55dbaf2b2d61f87d";
            };
          })
          (fetchCudaRedist {
            name = "libnvvm";
            version = "13.4.92";
            sha256 = {
              linux-x86_64 = "0e619cf3d5b27f7c812665599ca9761fc296e56129b24a85d822a5d8a1a3bd75";
              linux-sbsa = "483085a4f52d272ea2c5a6cb4727b24efc2cecab4c4df99e80dfc6548e5b7929";
            };
          })
          (fetchCudaRedist {
            name = "cuda_nvcc";
            version = "13.4.92";
            sha256 = {
              linux-x86_64 = "60998f40cc9df5826b2a846e2dcc328133fbc5ef539398ce0a21cf5dcac7193d";
              linux-sbsa = "8aebd451afb3984bc0019e803ae8a01a3837b7c69f36db779a7ce77e70831f5f";
            };
          })
          (fetchCudaRedist {
            name = "cuda_cudart";
            version = "13.4.92";
            sha256 = {
              linux-x86_64 = "0ac5dbc538d04e9983bc493b410cce4b459e1ee9f5f6654b6464ef7b3e14a8b5";
              linux-sbsa = "47e65e60d0f7cf832bff7992bd7a87ba4888195c31210392b5f23c607fe6c2a8";
            };
          })
          (fetchCudaRedist {
            name = "libcurand";
            version = "10.4.4.72";
            sha256 = {
              linux-x86_64 = "6f725f9187dc5f675308b830f786c163f5b1b499d86986f94fb60aa701c13c0b";
              linux-sbsa = "2968e2d86dce64896ee3c2ac86c9e0f7decec4a0f157c8d73e3d0f31582e9e0c";
            };
          })
          (fetchCudaRedist {
            name = "cuda_tileiras";
            version = "13.4.92";
            sha256 = {
              linux-x86_64 = "4b5f41dc0c99f2cb4dd20090c5a20a7756a6cfbf07c8dd21e960edec2e6ee3be";
              linux-sbsa = "3c88705c0d4f3fe2bd11d81acfb14f3cc43d6ee79d42767f9a244ba40e866027";
            };
          })
        ];
        # All archives must be extracted into one real directory tree, NOT
        # merged with symlinkJoin: tileiras locates libnvvm relative to its
        # own dereferenced path (/proc/self/exe), so through a symlink farm
        # it would look for nvvm/lib64/libnvvm.so inside the cuda_tileiras
        # store path, where it doesn't exist, and fail with exit status 5.
        cudaToolkit = pkgs.runCommand "cuda-toolkit-13.4.2" { } ''
          mkdir -p $out
          # --skip-old-files: cross-archive path collisions are first-wins
          # (see order note above) and show up in the build log instead of
          # silently taking whichever archive extracted last.
          ${pkgs.lib.concatMapStrings (src: ''
            tar xf ${src} --strip-components=1 -C $out --skip-old-files --warning=existing-file
          '') cudaRedistArchives}
          # cuda-bindings/build.rs also looks for libs in lib64/
          if [ -d "$out/lib" ] && [ ! -e "$out/lib64" ]; then
            ln -s lib "$out/lib64"
          fi
          # The exe-relative lookup above is the invariant that broke in
          # issue #172 — fail the build loudly if assembly ever loses it.
          if [ ! -x "$out/bin/tileiras" ]; then
            echo "cuda-toolkit assembly error: bin/tileiras missing or not executable" >&2
            exit 1
          fi
          if [ ! -f "$out/nvvm/lib64/libnvvm.so" ]; then
            echo "cuda-toolkit assembly error: nvvm/lib64/libnvvm.so missing (tileiras resolves it relative to its own binary)" >&2
            exit 1
          fi
        '';

        # bindgen uses libclang to parse CUDA headers.
        libclang = pkgs.llvmPackages_23.libclang;
        libcDev = pkgs.lib.getDev pkgs.stdenv.cc.libc;
        bindgenClangArgs = pkgs.lib.optionalString (!isDarwin) "-isystem ${libcDev}/include";

        # Stable Rust, matching the workspace and CI pin.
        rustToolchain =
          (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml).override
            {
              extensions = [
                "clippy"
                "rust-analyzer"
                "rust-src"
                "rustfmt"
              ];
            };
      in
      {
        devShells.default = pkgs.mkShell {
          hardeningDisable = [ "fortify" ];

          packages = [
            rustToolchain
            libclang
            pkgs.cmake
            pkgs.git
            pkgs.libffi
            pkgs.libxml2
            pkgs.ninja
            pkgs.pkg-config
            pkgs.python3
            pkgs.which
          ];

          CMAKE_GENERATOR = "Ninja";
          CUDA_TOOLKIT_PATH = "${cudaToolkit}";
          LIBCLANG_PATH = "${libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = bindgenClangArgs;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
            pkgs.libffi
            pkgs.libxml2
            libclang.lib
          ] ++ pkgs.lib.optionals (!isDarwin) [ cudaToolkit ]);

          shellHook = pkgs.lib.optionalString (!isDarwin) ''
            export PATH="${cudaToolkit}/bin:$PATH"
            # GPU driver libs: NixOS provides /run/opengl-driver/lib; on other
            # distros, symlink just the NVIDIA libs into a temp dir so we don't
            # pull in the host glibc.
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            else
              _nv_drv_dir=$(mktemp -d /tmp/nix-nvidia-driver.XXXXXX)
              # Search the host's own Debian multiarch dir (uname -m matches
              # the triplet prefix on x86_64 and aarch64), then generic dirs.
              for d in /usr/lib/$(uname -m)-linux-gnu /lib/$(uname -m)-linux-gnu /usr/lib /usr/lib64; do
                if [ -e "$d/libcuda.so.1" ]; then
                  for lib in "$d"/libcuda.so* "$d"/libnvidia-ptxjitcompiler.so* "$d"/libnvidia-gpucomp.so*; do
                    [ -e "$lib" ] && ln -sf "$lib" "$_nv_drv_dir/"
                  done
                  break
                fi
              done
              if [ -n "$(ls -A "$_nv_drv_dir" 2>/dev/null)" ]; then
                export LD_LIBRARY_PATH="$_nv_drv_dir:$LD_LIBRARY_PATH"
              else
                rm -rf "$_nv_drv_dir"
              fi
            fi
          '' + ''
            echo ""
            echo "cutile-rs dev shell"
            echo " ${if isDarwin then "~" else "✓"} CUDA  ${if isDarwin then "(headers only — no GPU required)" else "$CUDA_TOOLKIT_PATH"}"
            echo " ✓ Rust  $(rustc --version 2>/dev/null | awk '{print $2}')"
          '' + pkgs.lib.optionalString isDarwin ''
            echo ""
            echo "macOS: compile-only mode. Run:"
            echo "  cargo run -p cutile-examples --example compile_only"
          '' + ''
            echo ""
          '';
        };
      }
    );
}
