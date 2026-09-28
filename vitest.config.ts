import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts", "src/**/*.test.ts"],
    /**
     * golden.test.ts decodes every golden PNG of every fixture, and a page
     * carries ten full-size masks plus the autocropped, resized and
     * preprocessed pages: seconds of pngjs work, more on a machine that also
     * has local fixtures. The 5 s default is a budget for logic, not for that.
     */
    testTimeout: 30_000,
  },
});
