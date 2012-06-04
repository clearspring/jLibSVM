package libsvm.api;

import java.io.IOException;
import java.util.Arrays;

import libsvm.model.SVMEngine;
import libsvm.svm.model.SVMModel;
import libsvm.svm.model.FeatureNode;

public class ToySVMPredict
{
	private static final int RUNS = 1000;
	private SVMModel svmModel;
	private SVMEngine svmEngine;

	public ToySVMPredict(String svnModel) throws IOException
	{
		svmEngine = new SVMEngine();
		this.svmModel = svmEngine.svm_load_model(svnModel);
		System.out.println("SVM loaded !");
	}

	public double[] getPredicitionFor(FeatureNode[] nodes)
	{
		double[] results = new double[svmModel.nr_class];
		//svmEngine.svm_predict_probability(svmModel, nodes, results);
		if(svmEngine.svm_predict(svmModel, nodes)!=0)
			print(nodes);
		return results;
	}

	private void print(FeatureNode[] nodes)
	{
		for(FeatureNode node : nodes)
			System.out.println(node.getIndex()+":"+node.getValue()+", ");
		System.out.println();
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException
	{
		ToySVMPredict svmPredict = new ToySVMPredict(args[0]);
		
		int[] indexes = new int[]{568, 3162, 2783, 659, 3585, 1347, 694, 623, 142, 2913, 580, 594, 660, 718, 612, 3557, 1952, 685, 728, 596, 676, 868, 864, 606, 821, 686, 2359, 1474, 600, 135, 682, 715, 658, 2202, 3093, 591, 848, 696, 814, 747, 768, 578, 708, 833, 645, 556, 771, 3469, 804, 2472, 648, 598, 772, 557, 823, 3922};

		Arrays.sort(indexes);
		
		FeatureNode[] nodes = new FeatureNode[indexes.length];
		
		for(int i=0; i<indexes.length; i++)
		{
			nodes[i] = new FeatureNode(indexes[i], 1);
		}
		
		svmPredict.print(nodes);
		
		long start = System.currentTimeMillis();
		
		for (int i = 0; i < RUNS ; i++)
		{
			//double[] prob_estimates = new double[svmPredict.svmModel.nr_class];
			svmPredict.svmEngine.svm_predict(svmPredict.svmModel, nodes);
		}

		System.out.println("Runs : "+RUNS+" time secs/RUNS : "+((System.currentTimeMillis()-start)/1000.0)/RUNS);
	
	}

}
