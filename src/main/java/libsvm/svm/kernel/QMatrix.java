package libsvm.svm.kernel;


//
//Kernel evaluation
//
//the static method k_function is for doing single kernel evaluation
//the constructor of Kernel prepares to calculate the l*l kernel matrix
//the member function get_Q is for getting one column from the Q Matrix
//
public abstract class QMatrix {
	public abstract float[] get_Q(int column, int len);
	public abstract double[] get_QD();
	public abstract void swap_index(int i, int j);
};
