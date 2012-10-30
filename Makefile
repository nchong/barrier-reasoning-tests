default:
	cd clwrapper; make -f Makefile
	cd tests; make -f Makefile

clean:
	cd clwrapper; make -f Makefile clean
	cd tests; make -f Makefile clean
