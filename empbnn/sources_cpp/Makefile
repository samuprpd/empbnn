OL := -O3 -march=native
SRCS := main.cpp tensorlib/bnn.cpp

clang_fast_run: clean
	@clang++ $(OL) $(SRCS)
	@objdump -S a.out > a.s
	@./a.out > run_cpp.log

split_run: clean
	@clang++ $(OL) tensorlib/bnn.cpp main_split.cpp
	@objdump -S a.out > a.s
	@./a.out > run_cpp.log

clang_fast:
	@clang++ $(OL) $(SRCS)
	@objdump -S a.out > a.s

classic:
	@clang++ $(OL) main_classic.cpp
	@objdump -S a.out > a.s

split:
	@clang++ $(OL) tensorlib/bnn.cpp main_split.cpp
	@objdump -S a.out > a.s

clang: clean
	clang++ $(OL) -S -emit-llvm $(SRCS)
	clang++ $(OL) $(SRCS)
	objdump -S a.out > a.s
	clang++ $(OL) $(SRCS) -fsave-optimization-record
	llvm-opt-report main.opt.yaml -o main.lst
	opt -passes=dot-cfg-only main.ll -disable-output
	opt -passes=dot-callgraph main.ll -disable-output
	dot -Tpng -o main.cfg.png .main.dot 
	dot -Tpng -o main.clg.png main.ll.callgraph.dot
	./a.out > run_cpp.log

gcc: clean
	g++ $(OL) $(SRCS)
	./a.out
	objdump -S a.out > a.s

clean:
	@rm -f *.s *.ll *.out .*.dot *.dot *.png *.lst *.yaml *.log
