# Getting Started with Overleaf

This guide walks you through setting up the paper on Overleaf, which is a cloud based LaTeX editor that we recommend for collaborative academic writing. Overleaf handles the technical aspects of LaTeX compilation automatically, provides real time collaboration with your advisor and other contributors, maintains a complete version history of all changes, and allows you to access the document from any computer with a web browser. These features make it particularly well suited to academic research where multiple people may need to work on a document over months or years.

If you have not used Overleaf before, you can create a free account at overleaf.com using your university email address. The free tier supports projects with one or two collaborators and unlimited compile time, which is sufficient for our needs. Some universities including Florida have institutional Overleaf subscriptions that provide premium features at no cost to students, so it is worth checking with your institution before paying for an upgrade.

## Setting Up Your New Project

Once you have an Overleaf account, the first step is creating a new blank project. From the Overleaf homepage you will see a green button labeled New Project. Clicking this opens a menu where you should select Blank Project. Give your project a descriptive name like ICU Sedation Digital Twin Paper or whatever convention you and your advisor prefer for tracking research projects.

When the new project opens, Overleaf will create a default main.tex file with some sample content. You need to replace this default content with our actual paper file. The simplest way to do this is to delete the default main.tex contents entirely and paste in the contents of the main.tex file from our project repository. Open the file in any text editor, select all the content, copy it, then paste it into the Overleaf editor replacing what was there before.

Next you need to add the bibliography file. In the Overleaf editor, look for a button or menu option that says New File or has a plus icon for adding files. Create a new file named references.bib with that exact name and capitalization. Copy the contents of our references.bib file into this new file. Overleaf will recognize the file as a BibTeX bibliography and configure the compiler to process it automatically.

At this point you should be able to compile the document by clicking the green Recompile button at the top of the Overleaf editor. The first compilation may take longer than subsequent ones because Overleaf needs to install the packages we use. Once compilation completes you will see the rendered PDF in the right panel of the Overleaf interface. If compilation produces errors, the error messages will appear at the top of the right panel with line numbers and descriptions that should help you identify and fix any issues.

## Verifying Compiler Configuration

The references in our paper are processed using biber rather than the older bibtex tool, so we need to make sure Overleaf is configured to use biber. Click the menu icon in the top left of the Overleaf interface to open the project menu. Look for Compiler in the settings. The compiler should be set to pdfLaTeX which is the default. Look for an option called Settings or Bib tool. The bib processing tool should be set to biber rather than bibtex. If it shows bibtex, change it to biber. This setting is important because our references.bib file uses biblatex syntax which biber processes correctly while bibtex does not.

After verifying the compiler settings, recompile the document. The compilation should now process the bibliography correctly and produce a properly formatted PDF with numbered citations and a bibliography section at the end.

## Inviting Collaborators

To share the project with your advisor for collaborative editing, look for a Share button in the Overleaf interface. This opens a dialog where you can invite collaborators by their email address. The free tier of Overleaf allows one or two collaborators on each project depending on your subscription level. Your advisor will receive an email invitation and can accept it to gain access to the project.

Once your advisor has access, you can both edit the document simultaneously with changes appearing in real time. Overleaf indicates which user is editing which part of the document, so collaboration is generally smooth. If you both edit the same line at the same moment, Overleaf handles the conflict automatically by merging the changes when possible or prompting for resolution when not.

## Using Version History

One of the most valuable features of Overleaf for academic research is the version history. Every change you make is automatically saved with a timestamp, and you can return to any previous version at any time. To access version history, look for History in the top menu of the Overleaf interface. This opens a panel showing all the changes made to the project, with the ability to view the document as it existed at any point in time and restore previous versions if needed.

The version history is particularly useful when your advisor suggests changes that you implement, then later realize you preferred the original version. You can simply restore the version from before the changes rather than trying to manually reconstruct what was there. The history is also valuable for tracking the evolution of the document over time, which can be educational for understanding how academic writing develops through revision.

## Working with Figures

When the time comes to add figures to the paper, you upload them to Overleaf the same way you uploaded the bibliography file. Click the New File button and choose Upload, then select the figure file from your computer. The figures we generate from our analysis pipeline come in both PNG and PDF formats. PDF is generally preferred for journal submissions because it is a vector format that scales without quality loss, but PNG works fine for review purposes.

Reference figures in the paper text using the includegraphics command. For example, to include a figure named ablation_main.pdf that you uploaded to the project, you would write a figure environment in the LaTeX source like the following pattern. The actual code uses backslashes that I am avoiding here to keep this document readable, but you can find the correct syntax in the main.tex file when you start adding figures.

Figures should be referenced by name from the text using the ref command. For example, after adding a label to your figure environment, you can write something like "Figure ref{fig:ablation} shows the results of our ablation study." LaTeX automatically replaces the ref command with the correct figure number when compiling.

## Organizing Long Writing Sessions

Writing an academic paper takes weeks or months of work. To make this manageable, I recommend organizing your writing sessions around specific sections rather than trying to write everything at once. You might spend one session focused on the introduction, another on the methods, and so on. Within each session, set a goal for how much you want to accomplish so you have a clear endpoint that prevents the writing from feeling overwhelming.

Save your progress frequently by clicking the recompile button which both compiles the document and ensures the latest version is saved to the cloud. Overleaf saves automatically as you type, but recompiling provides explicit confirmation that everything is up to date.

When you finish a writing session, leave a comment for yourself or your advisor describing what you worked on and what should come next. Overleaf supports inline comments in the LaTeX source using the percent sign which marks the rest of a line as a comment that does not appear in the compiled document. These notes to your future self are invaluable when picking up writing again after a break.

## Compiling for Submission

When the paper is ready for submission, you need to download the compiled PDF and the LaTeX source files. From the Overleaf menu choose Download, then select either PDF for the compiled document or Source for a zip file containing all the LaTeX source code. Most journals want the PDF for review purposes and the source files for production after acceptance.

Some journals require specific formatting that may differ from our current setup. When we know which journal we are targeting, we can adjust the document class, page layout, and citation style to match that journal's requirements. Many journals provide their own LaTeX templates that we would integrate with our content at submission time. The key advantage of writing the paper in LaTeX from the beginning is that this template adaptation is much easier than converting from Word or Google Docs.

## Troubleshooting Common Issues

If compilation fails with errors about missing packages, the error message usually tells you which package is needed. Most LaTeX packages are available in Overleaf by default, but occasionally you may need to add a usepackage command in the document preamble for a package that is not currently included.

If citations show up as question marks instead of numbers, the bibliography has not been processed correctly. This usually means biber needs to run, which requires recompiling the document multiple times because LaTeX needs to know about the references on the first pass before biber can process them on the second pass. If you have set the bib tool to biber as described earlier, recompiling two or three times should resolve this.

If figures do not appear, verify that the figure files are uploaded to the project and that the file names match exactly what is referenced in the includegraphics commands. LaTeX is case sensitive on Linux servers like the ones Overleaf uses, so figure_one.pdf is different from Figure_One.pdf. Match capitalization exactly between the file name and the LaTeX reference.

If the document compiles but looks wrong with strange spacing or formatting, the issue is usually in the document class or one of the package configurations. The current setup uses the standard article class with reasonable defaults, so issues at this level are uncommon. If you encounter formatting problems that you cannot resolve, save your work and let me know what you are seeing and I can help troubleshoot.
