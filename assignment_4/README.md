# My template for LaTeX documents. 

It is based on the KOMA scripts and aims for documents with mathematical and software development topics. 

You can find an example in the current [releases](https://github.com/worldpotato/HM-LaTeX-Template/releases "releases").

# Build Status

[![Build Status](https://travis-ci.org/worldpotato/LaTeX-Template.svg?branch=master)](https://travis-ci.org/worldpotato/LaTeX-Template)

[![Latest Tag](https://badgen.net/github/tag/worldpotato/LaTeX-Template)](https://badgen.net/github/tag/worldpotato/LaTeX-Template)

# Structure

## Bibtex file

The bibtex file is located in `./bib/`

## Content

The `./content/` folder contains all the normal pages with your text and stuff.
All .tex files have a number as prefix in the order how they are created. 00_master.tex is the main .tex document where they all come together and 99 is the appendix.
I recomend to use a new .tex file for every chapter or logical part. They can easily be reordered by changing the order in the 00_master.tex file. If you need more chapters, your are doing something wrong.

## Images

In `./img/` are all graphics which are used in your document. You can create new folders in here.

## Preamble

One can find different stuff in the `./preamble/` folder. But mainly it's the background work. That means, you can edit the documentInfos.tex file to change the title page. And add special hyphenations or packages you need in here.

## Sourcecode

You can add codesnippets to your document by importing a sourcecode file. These files should be stored in here. Ofcourse you can create your own folders to organize the code. 
To import the sourcecode files has the advantage to be able to open the files in your favourite IDE and. 

# Vim

If vim is your favourite editor you can easily load the `vimSession.vim` file with `vim -S vimSession.vim` in the rood directory of that repository.
