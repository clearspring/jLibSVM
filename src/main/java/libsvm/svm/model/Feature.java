package libsvm.svm.model;

/**
 * @since 1.9
 */
public interface Feature extends Comparable<Feature>{

    int getIndex();

    double getValue();

    void setValue(double value);
}
