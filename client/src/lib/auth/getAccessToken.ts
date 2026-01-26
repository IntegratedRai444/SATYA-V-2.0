import { supabase } from "@/lib/supabaseSingleton"

export async function getAccessToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token ?? null
}

export async function getCurrentUser(): Promise<{ id: string; email: string } | null> {
  const { data } = await supabase.auth.getUser()
  return data.user ? { id: data.user.id, email: data.user.email || '' } : null
}
