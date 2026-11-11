# Uploading this repository to GitHub

## Before making the repository public

- Confirm that the raw/processed data, sample identifiers, manuscript, figures,
  and supplementary files may be redistributed.
- Replace the group author in `CITATION.cff` with the final author list if
  appropriate.
- Review `README.md`, especially the citation and contact sections.
- Run `python -m unittest discover -s tests -v`.

## Upload with Git

Create an empty GitHub repository named `FuseVirome`, then run these commands
from this directory:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/FuseVirome.git
git push -u origin main
```

Replace `YOUR-ACCOUNT` with the GitHub user or organisation name. If Git is not
available locally, extract the provided ZIP and use GitHub's web upload. Do not
upload the ZIP itself into the repository.
