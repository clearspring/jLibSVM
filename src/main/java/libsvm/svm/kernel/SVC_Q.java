package libsvm.svm.kernel;

import libsvm.svm.kernel.cache.Cache;
import libsvm.svm.model.SVMParams;
import libsvm.svm.model.SVNProblem;

//
// Q matrices for various formulations
//
public class SVC_Q extends Kernel
{
	private final byte[] y;
	private final Cache cache;
	private final double[] QD;

	public SVC_Q(SVNProblem prob, SVMParams param, byte[] y_)
	{
		super(prob.l, prob.x, param);
		y = (byte[])y_.clone();
		cache = new Cache(prob.l,(long)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = kernel_function(i,i);
	}

	public float[] get_Q(int i, int len)
	{
		float[][] data = new float[1][];
		int start, j;
		if((start = cache.get_data(i,data,len)) < len)
		{
			for(j=start;j<len;j++)
				data[0][j] = (float)(y[i]*y[j]*kernel_function(i,j));
		}
		return data[0];
	}

	public double[] get_QD()
	{
		return QD;
	}

	public void swap_index(int i, int j)
	{
		cache.swap_index(i,j);
		super.swap_index(i,j);
		do {byte _=y[i]; y[i]=y[j]; y[j]=_;} while(false);
		do {double _=QD[i]; QD[i]=QD[j]; QD[j]=_;} while(false);
	}
}
