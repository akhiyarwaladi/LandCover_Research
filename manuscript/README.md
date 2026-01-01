# Manuscript Folder - Scientific Writing with Claude

This folder is configured with **Scientific Writer** for writing academic papers about your land cover classification research.

## 📁 Folder Structure

```
manuscript/
├── CLAUDE.md          # Scientific Writer configuration (active in this folder)
├── README.md          # This file
├── figures/           # Paper figures (plots, diagrams, maps)
├── tables/            # Tables and supplementary data
├── references/        # BibTeX files and citations
└── drafts/            # Different versions of your manuscript
```

## 🚀 How to Use

### Writing Your Paper

When you work in this `manuscript/` folder, Claude will automatically use the Scientific Writer configuration and all its specialized skills:

**Example prompts:**
```
"Create a Remote Sensing of Environment journal paper on our land cover classification results"

"Write a Nature Scientific Reports manuscript about the Jambi Province land cover study"

"Generate an ISPRS journal paper with our Random Forest classification methodology"

"Create conference paper for IGARSS 2026 on this research"
```

### Available Skills

All 19+ Scientific Writer skills are available:
- **scientific-writing**: Full academic papers with IMRAD structure
- **research-lookup**: Real-time literature search for citations
- **citation-management**: BibTeX generation and reference management
- **venue-templates**: Journal-specific formatting (Nature, Science, RSE, etc.)
- **scientific-schematics**: Generate workflow diagrams, classification schemes
- **scientific-slides**: Create presentation slides for conferences
- **latex-posters**: Conference poster generation
- **peer-review**: Automatic manuscript review
- And many more...

### Quick Start

1. **Navigate to this folder** in your terminal:
   ```bash
   cd manuscript
   ```

2. **Start Claude Code in this folder** or specify this as your working directory

3. **Begin writing:**
   ```
   "Write a journal paper for Remote Sensing of Environment about our supervised
   land cover classification of Jambi Province using Random Forest and Sentinel-2
   imagery. Use the results from ../results/ and data described in ../CLAUDE.md"
   ```

### Tips

- **Figures**: The paper will reference figures from `figures/` - you can copy your plots from `../results/` here
- **References**: BibTeX files will be auto-generated in `references/`
- **Multiple Drafts**: Keep different versions in `drafts/` folder
- **Collaboration**: Each draft can be reviewed using the `peer-review` skill

### Recommended Journals

Based on your research (supervised land cover classification with Sentinel-2):

**Top Tier:**
- Remote Sensing of Environment
- ISPRS Journal of Photogrammetry and Remote Sensing
- IEEE Transactions on Geoscience and Remote Sensing

**High Impact Open Access:**
- Remote Sensing (MDPI)
- Scientific Reports (Nature)
- GIScience & Remote Sensing

**Regional Focus:**
- International Journal of Remote Sensing
- Journal of Applied Remote Sensing

### Data References

Your manuscript can reference:
- Project root: `../` (main CLAUDE.md has full project documentation)
- Results: `../results/` (classification outputs, figures, CSV)
- Code: `../modules/` and `../scripts/` (methodology implementation)
- Documentation: `../docs/` (research notes, issues, approaches)

---

**Ready to publish your research!** 🚀📄

Start with: `"Create a journal paper for [target journal] about our land cover classification research"`
