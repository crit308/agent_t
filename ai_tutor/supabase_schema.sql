-- Supabase Schema Initialization for AI Tutor Project
-- Run these commands in the Supabase SQL Editor.
-- Last Updated: 2024-04-07

-- ==========================================
-- 1. SESSIONS TABLE SETUP
-- ==========================================
-- Description: Stores the context and state for each user's tutoring session.

-- Create the table
CREATE TABLE public.sessions (
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL,
    context_data jsonb NOT NULL, -- Store the TutorContext object here
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    updated_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT sessions_pkey PRIMARY KEY (id),
    CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE CASCADE
);

-- Add comments
COMMENT ON TABLE public.sessions IS 'Stores AI tutor session state and context for users.';
COMMENT ON COLUMN public.sessions.id IS 'Unique identifier for the session.';
COMMENT ON COLUMN public.sessions.user_id IS 'Links to the authenticated user who owns the session.';
COMMENT ON COLUMN public.sessions.context_data IS 'JSONB blob containing the serialized TutorContext Pydantic model.';
COMMENT ON COLUMN public.sessions.created_at IS 'Timestamp when the session was created.';
COMMENT ON COLUMN public.sessions.updated_at IS 'Timestamp when the session was last updated.';

-- Create the trigger function for updated_at
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply the trigger to the sessions table
CREATE TRIGGER on_sessions_updated
BEFORE UPDATE ON public.sessions
FOR EACH ROW
EXECUTE PROCEDURE public.handle_updated_at();

-- Enable Row Level Security (RLS)
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

-- Grant basic permissions (usually handled by Supabase, but explicit grant is safe)
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.sessions TO authenticated;


-- ==========================================
-- 2. RLS POLICIES FOR sessions TABLE
-- ==========================================
-- Description: Ensures users can only access/modify their own session data.

-- Allow users to select their own sessions
CREATE POLICY "Allow individual user select access"
ON public.sessions
FOR SELECT
USING (auth.uid() = user_id);

-- Allow users to insert new sessions for themselves
CREATE POLICY "Allow individual user insert access"
ON public.sessions
FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- Allow users to update their own sessions
CREATE POLICY "Allow individual user update access"
ON public.sessions
FOR UPDATE
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

-- Allow users to delete their own sessions (Optional: Remove if deletion is not desired)
CREATE POLICY "Allow individual user delete access"
ON public.sessions
FOR DELETE
USING (auth.uid() = user_id);


-- ==========================================
-- 3. STORAGE BUCKET AND POLICIES
-- ==========================================
-- Description: Creates the bucket and sets policies for user-specific folder access.

-- Create Storage Bucket (assuming public access controlled by RLS)
INSERT INTO storage.buckets (id, name, public)
VALUES ('document_uploads', 'document_uploads', true)
ON CONFLICT (id) DO NOTHING; -- Avoid error if bucket already exists

-- RLS Policies for Storage Objects (within the 'document_uploads' bucket)
-- Assumes files are stored with a path like 'user_id/filename.pdf'

-- Allow users to view files in their own folder
CREATE POLICY "Allow user select access on own folder"
ON storage.objects
FOR SELECT
USING (
    bucket_id = 'document_uploads' AND
    auth.uid() = (storage.foldername(name))[1]::uuid -- checks the first part of the path
);

-- Allow users to insert files into their own folder
CREATE POLICY "Allow user insert access into own folder"
ON storage.objects
FOR INSERT
WITH CHECK (
    bucket_id = 'document_uploads' AND
    auth.uid() = (storage.foldername(name))[1]::uuid
);

-- Allow users to update files in their own folder
CREATE POLICY "Allow user update access in own folder"
ON storage.objects
FOR UPDATE
USING (
    bucket_id = 'document_uploads' AND
    auth.uid() = (storage.foldername(name))[1]::uuid
);

-- Allow users to delete files from their own folder
CREATE POLICY "Allow user delete access from own folder"
ON storage.objects
FOR DELETE
USING (
    bucket_id = 'document_uploads' AND
    auth.uid() = (storage.foldername(name))[1]::uuid
);

-- ==========================================
-- END OF SCHEMA INITIALIZATION
-- ========================================== 