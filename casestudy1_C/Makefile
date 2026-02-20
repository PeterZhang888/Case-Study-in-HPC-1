.PHONY: all case1 case1b compile1 compile1b submit1 submit1b submit clean

all: case1 case1b

# ---- Case 1: compile then submit ----
case1: compile1 submit1

compile1:
	./compile_casestudy1.sh

submit1: compile1
	sbatch submit_casestudy1.sh

# ---- Case 1b: compile then submit ----
case1b: compile1b submit1b

compile1b:
	./compile_casestudy1b.sh

submit1b: compile1b
	sbatch submit_casestudy1b.sh

submit: submit1 submit1b

clean:
	rm -f *.o casestudy1Exec casestudy1bExec *.out *.err

