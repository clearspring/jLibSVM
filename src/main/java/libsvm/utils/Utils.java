package libsvm.utils;

import libsvm.svm.model.FeatureNode;

public class Utils
{
	
	public static String prettyPrintNodes(FeatureNode[] nodes)
	{
		StringBuffer out = new StringBuffer();
		for(FeatureNode node : nodes)
		{
			out.append(node.index+",");
		}
		return out.toString();
	}
}
