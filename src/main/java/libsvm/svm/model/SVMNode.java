package libsvm.svm.model;

public class SVMNode implements Comparable<SVMNode>,java.io.Serializable
{
	private static final long serialVersionUID = -7706093303186855882L;
	public int index;
	public double value;

	public SVMNode(int index, double value)
	{
		super();
		this.index = index;
		this.value = value;
	}

	public SVMNode()
	{
		// TODO Auto-generated constructor stub
	}

	public int getIndex()
	{
		return index;
	}

	public void setIndex(int index)
	{
		this.index = index;
	}

	public double getValue()
	{
		return value;
	}

	public void setValue(double value)
	{
		this.value = value;
	}

	public int compareTo(SVMNode other)
	{
		return getIndex()<other.getIndex()?0:1;
	}

}
