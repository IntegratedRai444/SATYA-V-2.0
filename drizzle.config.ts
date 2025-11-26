import type { Config } from "drizzle-kit";

export default {
  schema: "./shared/schema.ts",
  out: "./server/db/migrations",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL || "postgresql://localhost:5432/satyaai"
  },
  migrations: {
    table: "migrations",
    schema: "public"
  }
} satisfies Config;
