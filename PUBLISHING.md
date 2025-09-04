# Publishing Guide for WASM Audio Ferrite

## Pre-Publication Checklist

- [ ] Update version numbers in:
  - [ ] `package.json`
  - [ ] `audio-processor-core/Cargo.toml`
  - [ ] `packages/toolkit/package.json`
- [ ] Update README with latest benchmarks
- [ ] Run all tests: `npm test && cd audio-processor-core && cargo test`
- [ ] Build release: `npm run build`
- [ ] Test demos locally
- [ ] Update CHANGELOG.md

## Publishing to NPM

```bash
# Login to NPM (first time only)
npm login

# Build the project
npm run build

# Dry run to check what will be published
npm publish --dry-run

# Publish to NPM
npm publish

# Tag the release
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

## Publishing to crates.io

```bash
# Navigate to Rust directory
cd audio-processor-core

# Login to crates.io (first time only)
cargo login

# Dry run
cargo publish --dry-run

# Publish
cargo publish

# Note: crates.io doesn't allow republishing same version
```

## Publishing to GitHub Packages

1. Create a `.npmrc` file in project root:
```
@yourusername:registry=https://npm.pkg.github.com
```

2. Authenticate with GitHub token:
```bash
npm login --registry=https://npm.pkg.github.com
# Username: YOUR_GITHUB_USERNAME
# Password: YOUR_GITHUB_TOKEN
```

3. Update package.json name to scoped:
```json
{
  "name": "@yourusername/wasm-audio-ferrite"
}
```

4. Publish:
```bash
npm publish --registry=https://npm.pkg.github.com
```

## CDN Access

Once published to NPM, automatically available at:
- unpkg: `https://unpkg.com/wasm-audio-ferrite@latest/dist/`
- jsDelivr: `https://cdn.jsdelivr.net/npm/wasm-audio-ferrite@latest/dist/`

## Installation Instructions for Users

### NPM/Yarn/PNPM
```bash
# NPM
npm install wasm-audio-ferrite

# Yarn
yarn add wasm-audio-ferrite

# PNPM
pnpm add wasm-audio-ferrite
```

### Rust/Cargo
```toml
[dependencies]
wasm-audio-ferrite = "0.1"
```

### Browser (CDN)
```html
<script type="module">
  import { NoiseProcessor } from 'https://unpkg.com/wasm-audio-ferrite@latest/dist/index.esm.js';
</script>
```

### Deno
```typescript
import { NoiseProcessor } from "https://unpkg.com/wasm-audio-ferrite@latest/dist/index.esm.js";
```

## Version Management

Follow semantic versioning:
- PATCH (0.0.x): Bug fixes, performance improvements
- MINOR (0.x.0): New features, backward compatible
- MAJOR (x.0.0): Breaking changes

## Post-Publication

1. Create GitHub Release with changelog
2. Update documentation site
3. Post announcement (Twitter, Reddit r/rust, r/webdev)
4. Update demo site with new version
5. Monitor GitHub issues for feedback

## Troubleshooting

### NPM Issues
- **E403**: Check npm login status
- **Name taken**: Add scope or choose different name
- **Files missing**: Check .npmignore and "files" in package.json

### Crates.io Issues
- **Version exists**: Bump version, can't republish same version
- **Size limit**: Check package size (10MB limit)
- **Invalid manifest**: Validate Cargo.toml

### WASM Build Issues
```bash
# Clean build
rm -rf target pkg dist
cargo clean
npm run build:wasm
```