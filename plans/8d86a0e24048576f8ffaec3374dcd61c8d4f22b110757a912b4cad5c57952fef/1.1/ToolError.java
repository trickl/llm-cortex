public class ToolError extends RuntimeException {
    public ToolError(String message) {
        super(message);
    }

    public ToolError(String message, Throwable cause) {
        super(message, cause);
    }
}
