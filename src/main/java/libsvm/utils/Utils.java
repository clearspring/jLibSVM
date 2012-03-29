package libsvm.utils;

import libsvm.svm.model.SVMNode;

public class Utils
{
	
	public static String prettyPrintNodes(SVMNode[] nodes)
	{
		StringBuffer out = new StringBuffer();
		for(SVMNode node : nodes)
		{
			out.append(node.index+",");
		}
		return out.toString();
	}
}
