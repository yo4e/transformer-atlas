import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const reference = z.object({
  label: z.string(),
  url: z.string().url()
});

const chapters = defineCollection({
  loader: glob({ base: "./src/content/chapters", pattern: "**/*.{md,mdx}" }),
  schema: z.object({
    title: z.string(),
    part: z.number().int().min(1).max(7),
    order: z.number().int().min(1).max(28),
    summary: z.string(),
    prerequisites: z.array(z.string()).default([]),
    concepts: z.array(z.string()).min(1),
    interactive_components: z.array(z.string()).default([]),
    references: z.array(reference).min(1),
    status: z.enum(["complete", "review-needed", "draft"]).default("complete")
  })
});

const math = defineCollection({
  loader: glob({ base: "./src/content/math", pattern: "**/*.md" }),
  schema: z.object({
    title: z.string(),
    order: z.number().int().min(1),
    summary: z.string(),
    concepts: z.array(z.string()).min(1),
    references: z.array(reference).min(1)
  })
});

export const collections = { chapters, math };
