/*
 * Decompiled with CFR 0_123.
 */
package opt.example;

import opt.EvaluationFunction;
import shared.Instance;
import util.linalg.Vector;

public class ContinuousPeaksEvaluationFunction implements EvaluationFunction {
    /**
     * The t value
     */
    private int t;
    public long fevals;
    
    /**
     * Make a new continuous peaks function
     * @param t the t value
     */
    public ContinuousPeaksEvaluationFunction(int t) {
        this.t = t;
    }

    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     */
    public double value(Instance d) {
        Vector data = d.getData();
        int max0 = 0;
        int count = 0;
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i) == 0) {
                count++;
            } else {
                if (count > max0) {
                    max0 = count;
                }
                count = 0;
            }
        }
        if (count > max0) {
            max0 = count;
        }
        int max1 = 0;
        count = 0;
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i) == 1) {
                count++;
            } else {
                if (count > max1) {
                    max1 = count;
                }
                count = 0;
            }
        }
        if (count > max1) {
            max1 = count;
        }
        int r = 0;
        if (max1 > t && max0 > t) {
            r = data.size();
        }
        ++this.fevals;
        return Math.max(max1, max0) + r;
    }
}

