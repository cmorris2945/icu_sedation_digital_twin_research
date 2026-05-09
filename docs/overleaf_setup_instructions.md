# Using the Paper Draft in Overleaf

This document explains how to set up the paper draft in Overleaf and use it effectively for collaborative writing with your advisor. Overleaf is a web-based LaTeX editor that lets multiple people work on the same document simultaneously, which is essential for academic paper writing where you need feedback from collaborators.

## Setting Up Your Overleaf Project

Begin by going to overleaf.com and creating an account if you do not already have one. The University of Florida likely has an institutional subscription to Overleaf Pro, which gives you additional features like full document history and unlimited collaborators. You should check whether you can access the Pro version through your university credentials before starting, since it makes collaboration much easier.

Once you are logged in, create a new blank project and give it a meaningful name like icu-sedation-paper or something similar that you will recognize easily later. Overleaf will create a default main.tex file with some sample content, which you can delete because we will replace it with our own content.

To get our paper draft into the project, you need to upload two files. First, upload the paper_draft.tex file from the docs directory of the repository I built for you. In Overleaf, click the upload button in the file panel on the left, then select the paper_draft.tex file. After it uploads, rename it to main.tex by right-clicking on the file name and selecting rename. The reason for this rename is that Overleaf expects the main document to be called main.tex by default, and using this convention avoids configuration headaches later.

Second, upload the references.bib file from the same docs directory. This file contains the bibliography that the paper draws from. Overleaf will automatically detect that it is a bibliography file and use it when compiling the document. You do not need to rename this file because the LaTeX preamble references it by the name references.

After uploading both files, click the green Recompile button to verify that the document compiles correctly. You should see the paper rendered on the right side of the screen with the title, author information, abstract, and section structure all properly formatted. If you see error messages instead, the most common issue is that Overleaf is missing one of the LaTeX packages we use. The error message will tell you which package is missing, and you can install most of them by adding them to the project through the menu options.

## Understanding the Document Structure

The document is organized into sections that correspond to the standard structure of medical journal articles. Each section is marked with a comment block in the LaTeX source explaining what that section should accomplish. As you write each section, you will replace the placeholder text with actual content while keeping the structural elements like section headings and subsection breaks.

Throughout the document, placeholder text appears in italics inside square brackets and is colored gray. This styling makes placeholders easy to find visually as you scroll through the document. The placeholders use specific language that hints at what should replace them, such as INSERT MAE VALUES HERE WITH 95 PERCENT CONFIDENCE INTERVALS rather than just TODO. This specificity helps you remember exactly what content needs to go in each location when you come back to fill it in.

The placeholders are produced by a custom command called placeholder that is defined in the preamble of the document. Before final submission, you can either remove this command definition entirely or redefine it to do nothing, which will eliminate all the placeholder text from the rendered output. During drafting, however, having the placeholders visible helps you track what is finished and what still needs work.

## Workflow for Collaborative Writing

The basic workflow for writing a paper in Overleaf with your advisor involves cycles of writing and feedback. You write a section or revise existing content, your advisor reads what you wrote and adds comments or suggested edits, you respond to those comments by revising further, and the cycle continues until the section is in good shape.

To enable this workflow, you need to share the project with your advisor through Overleaf's sharing feature. Click the share button at the top of the editor, then enter your advisor's email address. Overleaf will send them an invitation to collaborate on the project. Once they accept, both of you can edit the document and see each other's changes in real time.

During the drafting phase, you will probably want to use Overleaf's track changes feature, which is similar to the equivalent feature in Microsoft Word. With track changes enabled, edits made by either of you appear with author attribution and can be accepted or rejected by the other person. This makes it easy to maintain a record of what changed and who suggested each change. Track changes is available in Overleaf Pro accounts, which is another reason to verify your university subscription.

For commenting without making direct edits, you can use the comment feature to attach notes to specific text passages. Comments are useful when your advisor wants to suggest a change but is not certain about the right wording, or when you want to ask a question about how to handle a particular section. Comments persist in the document until they are explicitly resolved, providing a built-in way to track outstanding discussions.

## Compiling and Reviewing the Document

When you click the Recompile button, Overleaf processes the LaTeX source code through the LaTeX compiler and generates a PDF preview that appears on the right side of the screen. This preview shows you exactly how the final document will look, with all formatting, equations, tables, and references properly rendered.

If your changes do not appear in the preview after compiling, the most likely cause is a LaTeX error somewhere in the document. Look for the error indicator near the top of the preview pane, which will show how many errors were found. Click on the error indicator to see details about each error and the line number where it occurred. The most common errors are unmatched braces, undefined commands, and improperly escaped special characters in text.

For mathematical equations, verify that they render correctly by examining the preview after each significant change. LaTeX's mathematical typesetting is powerful but somewhat finicky, so equations that look correct in the source code can sometimes render in unexpected ways. The Eleveld pharmacokinetic equations in our methods section are particularly worth checking carefully because they contain many subscripts and superscripts that can be easy to get wrong.

## Filling In the Placeholders Over Time

The structure I created has many placeholders because the paper is being written before the experimental results are finalized. This is intentional and is actually a good way to write papers because it forces you to think about what each section needs to contain before you have the data to fill it in. Many researchers write papers in this skeleton-first style for exactly this reason.

As experimental results become available, you fill in the corresponding placeholders. The MAE values in the results tables get replaced with actual numbers from your analyses. The narrative descriptions in the results section get replaced with actual prose describing what the experiments showed. The discussion section gets filled in with actual interpretation based on the actual findings. By the time the paper is ready for submission, all placeholders have been replaced with content.

I recommend keeping the placeholder command active until very late in the writing process. Even when you have filled in most placeholders, you may still find a few that need attention, and having them visually highlighted makes them easy to find and complete.

## Adding Figures to the Document

Figures will be added once the experiments produce final visualizations. The standard workflow for adding a figure to a LaTeX document involves placing the image file in the project, then including it through the includegraphics command. In Overleaf, you upload the image file the same way you upload any other file, then add the appropriate LaTeX code to display it.

Each figure needs a caption that describes what is shown, a label for cross-referencing from the text, and proper placement parameters. The placement parameters tell LaTeX where to put the figure on the page, with options like h for here, t for top, b for bottom, and p for a separate page. For complex figures with multiple panels, you can use the subcaption package which is already loaded in our preamble.

When you add figures, place the image files in a subfolder called figures within the Overleaf project. This keeps the project organized and matches the file structure conventions used by most academic publishers. The figure references in the LaTeX source then use paths like figures/algorithm_comparison.pdf to locate the image files.

## Working With Tables

The paper template includes several tables that need to be filled in with actual data once experiments are complete. Tables in LaTeX use a specific syntax that can be intimidating at first but becomes natural with practice. The tables in our template use the booktabs package which produces high quality table formatting suitable for academic publications.

The basic structure of a table involves the tabular environment which specifies how many columns there are and how they should be aligned, then rows of data separated by double backslashes, with cells in each row separated by ampersands. The toprule, midrule, and bottomrule commands create the horizontal lines that separate the header from the data and frame the table.

When you fill in actual numbers from your experiments, replace the placeholder text but keep the structural elements like ampersands, double backslashes, and rule commands. If you accidentally remove these structural elements, the table will fail to compile or will render incorrectly.

## Submitting Drafts to Your Advisor

When you have a section or the full document ready for advisor review, you have two options for sharing it. The simpler option is to download the compiled PDF from Overleaf and email it as an attachment, which works well for advisors who prefer to review on their own time and provide consolidated feedback later. The more interactive option is to use Overleaf's built-in collaboration features, where your advisor can edit and comment directly in the document.

For early drafts, the email approach often works better because your advisor can read and think about the document without feeling pressured to respond immediately. For later drafts when you are converging on the final version, the collaborative approach is more efficient because changes can be made directly without back and forth emails.

Either way, when you submit a draft, include a brief note about what specifically you would like feedback on. Asking for general feedback typically produces less useful responses than asking specific questions like "I am concerned about the framing of the computational discovery in section three, do you think this approach makes the contribution clear enough?" Specific questions help your advisor focus their attention on the parts you most want input on.
