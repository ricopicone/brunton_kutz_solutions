.PHONY: main
	
title = Brunton and Kutz Problem 6.1 part d: Lorenz System Prediction
author = Rico A. R. Picone

main: main.docx # Default document(s) to make
	
%.txt: %.py
	python $< > $@

%.md: %.py
	publish $< md --title "$(title)" --author "$(author)"

%.tex: %.py
	publish $< tex --title "$(title)" --author "$(author)"
	
%.docx: %.py
	publish $< docx --title "$(title)" --author "$(author)"

%.pdf: %.py
	publish $< pdf --pdflatex True --title "$(title)" --author "$(author)"