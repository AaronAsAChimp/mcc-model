# Third-Party Notices

This project includes third-party works. Each keeps its own license; the MIT
license in [LICENSE](LICENSE) does not replace them.

## Icons — `web/public/team-icons/`

### Fluent UI Emoji (MIT)

The 32x32 team icons are derived from
[Fluent UI Emoji](https://github.com/microsoft/fluentui-emoji), recolored at
display time via CSS `currentColor`.

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Axolotl icon (CC BY 3.0)

`web/public/team-icons/aquaaxolotls.svg` is the "axolotl" icon from
[Game-icons.net](https://game-icons.net), created by **Caro Asercion**.

- Source: https://game-icons.net/1x1/caro-asercion/axolotl.html
- License: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)
- Changes: flattened fill to `currentColor` and removed the background square,
  following the same pattern as the other team icons.

## Typeface

`@fontsource/montserrat` bundles the Montserrat font, licensed under the
[SIL Open Font License 1.1](https://openfontlicense.org), by the Montserrat
Project Authors. The OFL text ships with the npm package; the font is not
relicensed by this project.

## Runtime dependencies

All bundled dependencies are permissively licensed (MIT, BSD-3-Clause,
Apache-2.0, OFL-1.1). Notably:

- Apache-2.0: `aiohttp`, `playwright`, `onnx`
- BSD-3-Clause: `numpy`, `torch`
- MIT: `astro`, `react`, `react-dom`, `onnxruntime`, `onnxruntime-web`,
  `onnxscript`, `beautifulsoup4`, `gql`, `plotly`

Each package ships its own license file in `node_modules/` or the installed
Python environment. If you redistribute a built artifact, keep those notices
intact.

## Trademarks

MC Championship is a tournament run by Noxcrew. This is an unofficial fan
project, not affiliated with or endorsed by Noxcrew. The project's licenses
grant no rights to any third-party names or marks.
