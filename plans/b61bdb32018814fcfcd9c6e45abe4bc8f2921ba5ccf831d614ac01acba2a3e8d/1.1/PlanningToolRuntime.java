import java.util.function.BiFunction;

public final class PlanningToolRuntime {
    private static BiFunction<String, Object[], Object> INVOKER;

    private PlanningToolRuntime() {}

    public static synchronized void setInvoker(BiFunction<String, Object[], Object> invoker) {
        INVOKER = invoker;
    }

    public static synchronized void clearInvoker() {
        INVOKER = null;
    }

    public static Object invoke(String toolName, Object... args) {
        BiFunction<String, Object[], Object> invoker = INVOKER;
        if (invoker == null) {
            throw new IllegalStateException("PlanningToolRuntime invoker not set");
        }
        return invoker.apply(toolName, args);
    }
}
