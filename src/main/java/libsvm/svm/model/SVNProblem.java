package libsvm.svm.model;

public class SVNProblem implements java.io.Serializable
{
	private static final long serialVersionUID = 7826924495622046474L;
	public int l;
	public double[] y;
	public SVMNode[][] x;
}
