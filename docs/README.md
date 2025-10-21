# Documentation

This directory contains comprehensive documentation for the CellMorphNet project.

## Documentation Files

### Core Documentation

#### `GETTING_STARTED.md`
**Purpose**: Quick start guide for new users

**Contents**:
- Installation instructions
- First model training
- Running inference
- Using the web interface

**Target Audience**: New users, developers getting started

---

#### `DATASETS.md`
**Purpose**: Complete guide to datasets used in the project

**Contents**:
- BloodMNIST dataset overview
- BCCD dataset details
- LISC database information
- Dataset preparation instructions
- Data format specifications
- Download and preprocessing guides

**Target Audience**: Researchers, data scientists

---

#### `TRAINING_ANALYSIS.md`
**Purpose**: In-depth analysis of model training performance

**Contents**:
- Training metrics and curves
- Model architecture comparisons
- Hyperparameter tuning results
- Performance benchmarks
- Per-class accuracy analysis
- Training optimization strategies

**Target Audience**: ML engineers, researchers

---

#### `TRAINING_SUMMARY.md`
**Purpose**: Concise summary of training configurations and results

**Contents**:
- Best model configurations
- Key hyperparameters
- Performance summary tables
- Quick reference guide

**Target Audience**: Quick reference for all users

---

### Deployment and Usage

#### `DEPLOYMENT_GUIDE.md`
**Purpose**: Complete deployment instructions

**Contents**:
- Streamlit web app deployment
- FastAPI server setup
- Model export (CoreML, ONNX, TorchScript)
- Production deployment strategies
- Docker containerization
- Cloud deployment options

**Target Audience**: DevOps, production deployment teams

---

#### `HOW_TO_TEST.md`
**Purpose**: Guide for properly testing the model

**Contents**:
- Testing methodologies
- Image format requirements
- Proper test image preparation
- Why screenshots may not work
- Best practices for model evaluation

**Target Audience**: Users, QA teams

---

### Troubleshooting and Fixes

#### `SCREENSHOT_FIX_GUIDE.md`
**Purpose**: Detailed explanation of screenshot classification issues

**Contents**:
- Root cause analysis of scale mismatch
- Preprocessing strategies comparison
- Downscale fix implementation
- Usage instructions
- Visual examples and comparisons

**Target Audience**: Users encountering classification issues

---

### Project Information

#### `PROJECT_SUMMARY.md`
**Purpose**: High-level project overview and achievements

**Contents**:
- Project goals and motivation
- Key features and capabilities
- Performance highlights
- Technology stack
- Future roadmap

**Target Audience**: Stakeholders, managers, new contributors

---

#### `overview.md`
**Purpose**: Technical architecture overview

**Contents**:
- System architecture
- Component descriptions
- Data flow diagrams
- Technical specifications
- Integration points

**Target Audience**: Technical leads, architects

---

#### `README_old.md`
**Purpose**: Previous version of main README (archived)

**Contents**: Original project documentation with emoji-rich formatting

**Note**: Kept for reference; replaced by professional README.md

---

## Documentation Organization

```
docs/
├── README.md                    # This file
├── GETTING_STARTED.md          # Quick start guide
├── DATASETS.md                 # Dataset documentation
├── TRAINING_ANALYSIS.md        # Training performance analysis
├── TRAINING_SUMMARY.md         # Training quick reference
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
├── HOW_TO_TEST.md              # Testing guide
├── SCREENSHOT_FIX_GUIDE.md     # Troubleshooting guide
├── PROJECT_SUMMARY.md          # Project overview
├── overview.md                 # Technical architecture
└── README_old.md               # Archived original README
```

---

## Reading Path

### For New Users
1. Start with [GETTING_STARTED.md](GETTING_STARTED.md)
2. Review [DATASETS.md](DATASETS.md) to understand data requirements
3. Follow [HOW_TO_TEST.md](HOW_TO_TEST.md) for testing guidelines
4. Read [SCREENSHOT_FIX_GUIDE.md](SCREENSHOT_FIX_GUIDE.md) if encountering issues

### For Developers
1. Read [overview.md](overview.md) for architecture
2. Study [TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md) for model details
3. Review [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment
4. Consult [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) for quick reference

### For Researchers
1. Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for context
2. Study [DATASETS.md](DATASETS.md) for data specifications
3. Analyze [TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md) for methodology
4. Reference main [README.md](../README.md) for citations

---

## Documentation Standards

### Writing Style
- Clear and concise language
- Professional tone without emojis
- Code examples with syntax highlighting
- Step-by-step instructions where applicable

### Structure
- Start with purpose and target audience
- Include table of contents for long documents
- Use headers for easy navigation
- Provide examples and use cases

### Code Examples
- Use bash code blocks for terminal commands
- Use python code blocks for Python code
- Include expected outputs where helpful
- Show both simple and advanced examples

### Updates
- Update documentation when features change
- Keep version information current
- Archive old documentation rather than deleting
- Date major updates

---

## Contributing to Documentation

When adding or updating documentation:

1. **Choose the right file**: Use existing files for related content
2. **Follow the structure**: Match the style of existing docs
3. **Be comprehensive**: Include all necessary details
4. **Add examples**: Provide practical usage examples
5. **Update this README**: Add new docs to the table above
6. **Test instructions**: Verify all commands and code work

### New Documentation Template

```markdown
# Document Title

Brief description of what this document covers.

## Purpose

Explain why this document exists and what problem it solves.

## Target Audience

Who should read this document.

## Contents

[Table of contents]

## Main Content

[Detailed content with examples]

## Related Documentation

- Link to related docs
- Link to main README

---

For questions or suggestions, see [README.md](../README.md) for contact information.
```

---

## Additional Resources

### Main Project Files
- [README.md](../README.md) - Main project README
- [requirements.txt](../requirements.txt) - Python dependencies
- [LICENSE](../LICENSE) - Project license

### Code Documentation
- Source code in `src/` with inline comments
- Script documentation in `scripts/`
- Demo applications in `demos/`
- Test scripts in `tests/`

### External Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MedMNIST Paper](https://medmnist.com/)

---

## Documentation Maintenance

### Review Schedule
- **Monthly**: Check for outdated information
- **Per Release**: Update version-specific details
- **As Needed**: Fix errors and clarify confusion

### Feedback
If you find issues in documentation:
1. Open an issue on GitHub
2. Suggest improvements
3. Submit pull requests with corrections

---

**Last Updated**: October 2025

For the latest updates, see the [GitHub repository](https://github.com/parth-6-5-4/CellMorphNet).
