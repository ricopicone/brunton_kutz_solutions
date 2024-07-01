.PHONY: main
	
title = Brunton and Kutz Problem 4.3: Polynomial Regression with Regularization
author = Rico A. R. Picone

main: main.pdf # Default document(s) to make
	
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