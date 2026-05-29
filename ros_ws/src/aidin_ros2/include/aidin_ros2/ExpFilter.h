class ExponentialFilter {
private:
    float alpha;  // Smoothing factor
    float output; // Current output of the filter

public:
    // Constructor to initialize the smoothing factor and initial output
    ExponentialFilter(float alpha, float initial_output = 0.0f)
        : alpha(alpha), output(initial_output) {}

    // Method to apply the filter to a new input value
    float apply(float input) {
        // Update output using the exponential filter formula
        output = alpha * input + (1.0f - alpha) * output;
        return output;
    }

    // Getter for the current output
    float getOutput() const {
        return output;
    }

    // Setter for the alpha value (smoothing factor)
    void setAlpha(float newAlpha) {
        alpha = newAlpha;
    }

    // Getter for the alpha value (smoothing factor)
    float getAlpha() const {
        return alpha;
    }
};

