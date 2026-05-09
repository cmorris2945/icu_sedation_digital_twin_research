# Paper Directory

This directory contains the LaTeX source for the paper that will eventually be submitted to a medical informatics journal. The paper is being drafted in LaTeX from the beginning rather than being written in Word and converted later, because LaTeX produces the typesetting quality that medical journals expect, handles mathematical notation and citations far better than word processors, and integrates naturally with reference management workflows.

The two main files in this directory work together to produce the final paper. The main.tex file contains the document structure, all the prose content, and references to figures and tables that are stored elsewhere in the project. The references.bib file contains all the bibliographic entries in BibTeX format, which the biblatex package processes automatically to generate properly formatted citations and the bibliography section. When you compile main.tex, LaTeX combines these two files along with any included figures to produce a PDF document.

The current state of the document is a complete skeleton with the section structure, abstract framework, and substantial placeholder content describing what each section will eventually contain. The bibliography is partially populated with the references we have already identified as essential, with placeholders for additional references that will be added as the writing progresses. As actual experimental results become available, the placeholder content in the methods and results sections will be replaced with detailed descriptions of what we did and what we found.

The paper is being written for medical informatics journals like NPJ Digital Medicine and the British Journal of Anaesthesia, which expect structured abstracts, detailed methods sections sufficient for reproduction, and clear separation between results and interpretation. The structure of the document reflects these conventions. When we know which specific journal we are targeting we can adjust the document class and packages to match that journal's specific template requirements.

## Compiling the Paper

The recommended way to compile this paper is through Overleaf, which is a cloud based LaTeX editor that handles compilation automatically and provides real time collaboration features useful for working with your advisor. To set up the paper in Overleaf, create a new blank project, then upload the main.tex and references.bib files from this directory. Overleaf will detect the document class and packages used and configure the compiler appropriately. The default biber compiler that Overleaf uses for biblatex is exactly what our references.bib setup expects, so no additional configuration should be needed.

If you prefer to compile locally rather than using Overleaf, you need a LaTeX distribution installed on your computer. The most common distributions are TeX Live for Linux and Mac, MiKTeX for Windows, and MacTeX for Mac specifically. Once installed, compile the document by running pdflatex main.tex, then biber main, then pdflatex main.tex twice more. The multiple passes are needed because LaTeX needs to know about cross references and the bibliography needs to be processed separately. The repeated runs ensure that all references resolve correctly. Most LaTeX editors handle this multi step compilation automatically.

## Working with the Bibliography

The references.bib file uses BibTeX format which is the standard way to store bibliographic information for LaTeX documents. Each entry has a type indicating what kind of work it is (article, book, inproceedings for conference papers, and so on), a key that you use to cite it from the main text, and fields containing the actual bibliographic information.

When you cite a reference in the paper text, you use the cite command followed by the reference key. For example, citing the Eleveld 2018 paper looks like \cite{eleveld_2018} which produces a numbered citation in the final document. The biblatex package handles all the formatting automatically based on the citation style we configured in the document preamble.

To add a new reference, paste a properly formatted BibTeX entry into references.bib in the appropriate category section. You can usually copy BibTeX entries directly from publisher websites, Google Scholar, or reference management software like Zotero or Mendeley. Verify the entry format and field accuracy before relying on it because automatically generated entries sometimes have errors or formatting issues that need fixing.

## Document Structure

The document is organized into sections that follow conventions for medical informatics papers. The introduction establishes the clinical importance of the problem, reviews prior work, and states our specific contributions and hypotheses. The methods section describes our data source, mechanistic model, machine learning algorithms, ablation study design, and statistical analysis approach in enough detail that another researcher could reproduce our work. The results section presents our empirical findings without interpretation, including tables and figures showing the actual numbers we obtained. The discussion section interprets the results, places them in context with prior literature, acknowledges limitations, and suggests future directions. The conclusion briefly summarizes the take home messages.

Each section begins with a top level header that organizes the content. Subsections within each section provide finer grained organization. The section structure visible in the table of contents that LaTeX generates automatically will give readers a clear roadmap through the document.

## Style Conventions

The writing uses formal academic prose appropriate for medical journals. Specific style conventions we follow include using past tense for describing what we did and what we found, present tense for describing what is true generally, and future tense sparingly for describing planned work. Acronyms are spelled out on first use with the abbreviation in parentheses. Technical terms are introduced with definitions when they may be unfamiliar to readers from adjacent fields. Statistical reporting follows medical conventions with point estimates, confidence intervals when appropriate, and explicit reporting of sample sizes.

The document uses biblatex with numeric citation style which is the most common style in medical informatics journals. References appear as numbers in square brackets in the text, with the full bibliography listed at the end of the document in order of first appearance.
