-- 1. セキュリティ強化（RLSの有効化）
alter table public.trip_memories enable row level security;

-- 2. MVP用の一時的なパブリック許可ポリシー
-- （※次の段階でAuth/ログイン機能を入れるまでの暫定措置です）
create policy "trip_memories_public_read"
on public.trip_memories
for select to anon using (true);

create policy "trip_memories_public_insert"
on public.trip_memories
for insert to anon with check (true);

-- 3. 検索関数のアップグレード（Threshold・足切り機能の追加）
-- 引数の数が変わるため、一度古い関数を削除します
drop function if exists public.match_memories(vector, int);

create or replace function public.match_memories (
  query_embedding vector(768),
  match_threshold float,
  match_count int
)
returns table (
  id uuid, "date" text, location text, text_memo text,
  image_paths jsonb, audio_path text, video_path text, pdf_path text,
  "timestamp" text, similarity float
)
language sql stable
as $$
  select
    id, "date", location, text_memo, image_paths, audio_path, video_path, pdf_path, "timestamp",
    1 - (embedding <=> query_embedding) as similarity
  from public.trip_memories
  -- ★ここが重要：類似度がthreshold（閾値）より高いものだけを残す
  where embedding <=> query_embedding < 1 - match_threshold
  order by embedding <=> query_embedding asc
  limit least(match_count, 200);
$$;