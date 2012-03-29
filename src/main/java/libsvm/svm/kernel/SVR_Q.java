package libsvm.svm.kernel;

import libsvm.svm.kernel.cache.Cache;
import libsvm.svm.model.SVMParams;
import libsvm.svm.model.SVNProblem;

public class SVR_Q extends Kernel
{
	private final int l;
	private final Cache cache;
	private final byte[] sign;
	private final int[] index;
	private int next_buffer;
	private float[][] buffer;
	private final double[] QD;

	public SVR_Q(SVNProblem prob, SVMParams param)
	{
		super(prob.l, prob.x, param);
		l = prob.l;
		cache = new Cache(l,(long)(param.cache_size*(1<<20)));
		QD = new double[2*l];
		sign = new byte[2*l];
		index = new int[2*l];
		for(int k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k] = kernel_function(k,k);
			QD[k+l] = QD[k];
		}
		buffer = new float[2][2*l];
		next_buffer = 0;
	}

	public void swap_index(int i, int j)
	{
		do {byte _=sign[i]; sign[i]=sign[j]; sign[j]=_;} while(false);
		do {int _=index[i]; index[i]=index[j]; index[j]=_;} while(false);
		do {double _=QD[i]; QD[i]=QD[j]; QD[j]=_;} while(false);
	}

	public float[] get_Q(int i, int len)
	{
		float[][] data = new float[1][];
		int j, real_i = index[i];
		if(cache.get_data(real_i,data,l) < l)
		{
			for(j=0;j<l;j++)
				data[0][j] = (float)kernel_function(real_i,j);
		}

		// reorder and copy
		float buf[] = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		byte si = sign[i];
		for(j=0;j<len;j++)
			buf[j] = (float) si * sign[j] * data[0][index[j]];
		return buf;
	}

	public double[] get_QD()
	{
		return QD;
	}
}
