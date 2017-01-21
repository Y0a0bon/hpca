###############
##  Makefile ##
###############

SUBDIRS := $(wildcard */.)

all :
	@$(MAKE) -C projet_partie_1
	@$(MAKE) -C projet_partie_2

exec :
	@$(MAKE) -C projet_partie_1 exec
	@$(MAKE) -C projet_partie_2 exec

clean :
	@$(MAKE) -C projet_partie_1 clean
	@$(MAKE) -C projet_partie_2 clean
