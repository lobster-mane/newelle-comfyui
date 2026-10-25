{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  name = "newelle-comfyui";

  packages = [
    (pkgs.python3.withPackages (ps: with ps; [
      pillow
      pygobject3
      requests
    ]))
    pkgs.gobject-introspection
    pkgs.ruff
  ];

  PYTHONPATH = "";

  shellHook = ''
    echo "newelle-comfyui development shell"
    echo ""
    echo "  Lint:  ruff check comfyui.py tests/"
    echo "  Test:  python tests/mock_comfyui.py"
  '';
}