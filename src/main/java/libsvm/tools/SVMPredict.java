package libsvm.tools;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

import libsvm.model.SVMEngine;
import libsvm.svm.model.SVMModel;
import libsvm.svm.model.SVMNode;
import libsvm.svm.model.SVMParams;
import libsvm.utils.Utils;

public class SVMPredict
{
	
	private static double atof(String s)
	{
		return Double.valueOf(s).doubleValue();
	}

	private static  int atoi(String s)
	{
		return Integer.parseInt(s);
	}

	private void predict(BufferedReader input, DataOutputStream output, SVMModel model, int predict_probability,SVMEngine svmInstance) throws IOException
	{
		int correct = 0;
		int total = 0;
		double error = 0;
		double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

		int svm_type=svmInstance.svm_get_svm_type(model);
		int nr_class=svmInstance.svm_get_nr_class(model);
		double[] prob_estimates=null;

		if(predict_probability == 1)
		{
			if(svm_type == SVMParams.EPSILON_SVR ||
			   svm_type == SVMParams.NU_SVR)
			{
				System.out.print("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma="+svmInstance.svm_get_svr_probability(model)+"\n");
			}
			else
			{
				int[] labels=new int[nr_class];
				svmInstance.svm_get_labels(model,labels);
				prob_estimates = new double[nr_class];
				output.writeBytes("labels");
				for(int j=0;j<nr_class;j++)
					output.writeBytes(" "+labels[j]);
				output.writeBytes("\n");
			}
		}
		while(true)
		{
			String line = input.readLine();
			if(line == null) break;

			StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");

			double target = atof(st.nextToken());
			int m = st.countTokens()/2;
			SVMNode[] x = new SVMNode[m];
			for(int j=0;j<m;j++)
			{
				x[j] = new SVMNode();
				x[j].index = atoi(st.nextToken());
				x[j].value = atof(st.nextToken());
			}

			print(x);
			
			double v;
			if (predict_probability==1 && (svm_type== SVMParams.C_SVC || svm_type== SVMParams.NU_SVC))
			{
				v = svmInstance.svm_predict_probability(model,x,prob_estimates);
				output.writeBytes(v+" ");
				for(int j=0;j<nr_class;j++)
					output.writeBytes(prob_estimates[j]+" ");
				output.writeBytes("\n");
			}
			else
			{
				v = svmInstance.svm_predict(model,x);
				if(v!=0)
				{
					System.out.println(Utils.prettyPrintNodes(x));
					System.exit(0);
				}
				//output.writeBytes(v+"\n");
			}

			if(v == target)
				++correct;
			error += (v-target)*(v-target);
			sumv += v;
			sumy += target;
			sumvv += v*v;
			sumyy += target*target;
			sumvy += v*target;
			++total;
		}
		if(svm_type == SVMParams.EPSILON_SVR ||
		   svm_type == SVMParams.NU_SVR)
		{
			System.out.print("Mean squared error = "+error/total+" (regression)\n");
			System.out.print("Squared correlation coefficient = "+
				 ((total*sumvy-sumv*sumy)*(total*sumvy-sumv*sumy))/
				 ((total*sumvv-sumv*sumv)*(total*sumyy-sumy*sumy))+
				 " (regression)\n");
		}
		else
			System.out.print("Accuracy = "+(double)correct/total*100+
				 "% ("+correct+"/"+total+") (classification)\n");
	}

	private static void print(SVMNode[] nodes)
	{
		for(SVMNode node : nodes)
			System.out.println(node.getIndex()+":"+node.getValue()+", ");
		System.out.println();
	}
	
	private static  void exit_with_help()
	{
		System.err.print("usage: SVMPredict [options] test_file model_file output_file\n"
		+"options:\n"
		+"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n");
		System.exit(1);
	}

	public static void main(String argv[]) throws IOException
	{
		SVMPredict predictor = new SVMPredict();
		int i, predict_probability=0;
		SVMEngine svmInstance = new SVMEngine();
		
		// parse options
		for(i=0;i<argv.length;i++)
		{
			if(argv[i].charAt(0) != '-') break;
			++i;
			switch(argv[i-1].charAt(1))
			{
				case 'b':
					predict_probability = atoi(argv[i]);
					break;
				default:
					System.err.print("Unknown option: " + argv[i-1] + "\n");
					exit_with_help();
			}
		}
		if(i>=argv.length-2)
			exit_with_help();
		try 
		{
			BufferedReader input = new BufferedReader(new FileReader(argv[i]));
			DataOutputStream output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(argv[i+2])));
			SVMModel model = svmInstance.svm_load_model(argv[i+1]);
			if(predict_probability == 1)
			{
				if(svmInstance.svm_check_probability_model(model)==0)
				{
					System.err.print("Model does not support probabiliy estimates\n");
					System.exit(1);
				}
			}
			else
			{
				if(svmInstance.svm_check_probability_model(model)!=0)
				{
					System.out.print("Model supports probability estimates, but disabled in prediction.\n");
				}
			}
			predictor.predict(input,output,model,predict_probability,svmInstance);
			input.close();
			output.close();
		} 
		catch(FileNotFoundException e) 
		{
			e.printStackTrace();
			exit_with_help();
		}
		catch(ArrayIndexOutOfBoundsException e) 
		{
			e.printStackTrace();
			exit_with_help();
		}
	}
}
