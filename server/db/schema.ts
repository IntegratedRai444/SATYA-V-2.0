import { pgTable, serial, text, varchar, timestamp, boolean, jsonb, integer, primaryKey, pgEnum, uuid, inet } from 'drizzle-orm/pg-core';
import { createInsertSchema, createSelectSchema } from 'drizzle-zod';
import { z } from 'zod';

// Enums
export const userRoleEnum = pgEnum('user_role', ['user', 'admin', 'moderator']);
export const providerEnum = pgEnum('provider', ['email', 'google', 'github', 'microsoft', 'apple']);

export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  username: varchar('username', { length: 50 }).notNull().unique(),
  email: varchar('email', { length: 255 }).unique(),
  password: varchar('password', { length: 255 }),
  fullName: varchar('full_name', { length: 100 }),
  role: userRoleEnum('role').notNull().default('user'),
  provider: providerEnum('provider').notNull().default('email'),
  providerId: varchar('provider_id', { length: 255 }),
  avatarUrl: text('avatar_url'),
  isVerified: boolean('is_verified').notNull().default(false),
  verificationToken: varchar('verification_token', { length: 255 }),
  verificationExpires: timestamp('verification_expires', { withTimezone: true }),
  resetToken: varchar('reset_token', { length: 255 }),
  resetExpires: timestamp('reset_expires', { withTimezone: true }),
  lastLogin: timestamp('last_login', { withTimezone: true }),
  loginCount: integer('login_count').notNull().default(0),
  failedLoginAttempts: integer('failed_login_attempts').notNull().default(0),
  lastFailedLogin: timestamp('last_failed_login', { withTimezone: true }),
  lastPasswordChange: timestamp('last_password_change', { withTimezone: true }),
  lastIp: inet('last_ip'),
  timezone: varchar('timezone', { length: 50 }).default('UTC'),
  locale: varchar('locale', { length: 10 }).default('en-US'),
  isActive: boolean('is_active').notNull().default(true),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
  deletedAt: timestamp('deleted_at', { withTimezone: true })
});

export const sessions = pgTable('sessions', {
  id: uuid('id').defaultRandom().primaryKey(),
  userId: integer('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  refreshToken: text('refresh_token').notNull(),
  userAgent: text('user_agent'),
  ipAddress: inet('ip_address'),
  expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
  isRevoked: boolean('is_revoked').notNull().default(false),
  revokedAt: timestamp('revoked_at', { withTimezone: true }),
  lastUsedAt: timestamp('last_used_at', { withTimezone: true }).notNull().defaultNow(),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow()
});

export const oauthStates = pgTable('oauth_states', {
  state: text('state').primaryKey(),
  provider: providerEnum('provider').notNull(),
  codeVerifier: text('code_verifier').notNull(),
  redirectUri: text('redirect_uri').notNull(),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  expiresAt: timestamp('expires_at', { withTimezone: true }).notNull()
});

export const revokedTokens = pgTable('revoked_tokens', {
  jti: text('jti').primaryKey(),
  userId: integer('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
  revokedAt: timestamp('revoked_at', { withTimezone: true }).notNull().defaultNow(),
  reason: text('reason')
});

export const userPreferences = pgTable('user_preferences', {
  id: serial('id').primaryKey(),
  userId: integer('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  emailNotifications: boolean('email_notifications').notNull().default(true),
  pushNotifications: boolean('push_notifications').notNull().default(true),
  twoFactorEnabled: boolean('two_factor_enabled').notNull().default(false),
  preferredProvider: providerEnum('preferred_provider'),
  theme: varchar('theme', { length: 20 }),
  notifications: jsonb('notifications').default({}).notNull(),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow()
});

export const auditLogs = pgTable('audit_logs', {
  id: serial('id').primaryKey(),
  userId: integer('user_id').references(() => users.id, { onDelete: 'set null' }),
  action: varchar('action', { length: 100 }).notNull(),
  entityType: varchar('entity_type', { length: 50 }),
  entityId: integer('entity_id'),
  ipAddress: inet('ip_address'),
  userAgent: text('user_agent'),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow()
});

// Zod schemas for validation
export const insertUserSchema = createInsertSchema(users, {
  email: (schema) => schema.email({ message: 'Invalid email address' }),
  username: (schema) => schema.min(3, 'Username must be at least 3 characters')
    .max(50, 'Username cannot be longer than 50 characters')
    .regex(/^[a-zA-Z0-9_]+$/, 'Username can only contain letters, numbers and underscores'),
  password: (schema) => schema.min(8, 'Password must be at least 8 characters')
    .max(100, 'Password cannot be longer than 100 characters')
});

export const selectUserSchema = createSelectSchema(users);
export type User = z.infer<typeof selectUserSchema>;

export const insertSessionSchema = createInsertSchema(sessions);
export const selectSessionSchema = createSelectSchema(sessions);
export type Session = z.infer<typeof selectSessionSchema>;

export const insertOAuthStateSchema = createInsertSchema(oauthStates);
export const selectOAuthStateSchema = createSelectSchema(oauthStates);
export type OAuthState = z.infer<typeof selectOAuthStateSchema>;

export const insertRevokedTokenSchema = createInsertSchema(revokedTokens);
export const selectRevokedTokenSchema = createSelectSchema(revokedTokens);
export type RevokedToken = z.infer<typeof selectRevokedTokenSchema>;

export const insertUserPreferenceSchema = createInsertSchema(userPreferences);
export const selectUserPreferenceSchema = createSelectSchema(userPreferences);
export type UserPreference = z.infer<typeof selectUserPreferenceSchema>;

export const insertAuditLogSchema = createInsertSchema(auditLogs);
export const selectAuditLogSchema = createSelectSchema(auditLogs);
export type AuditLog = z.infer<typeof selectAuditLogSchema>;
