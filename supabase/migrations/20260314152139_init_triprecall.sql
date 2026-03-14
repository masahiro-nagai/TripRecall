-- ① AIベクトル検索用の拡張機能(pgvector)を有効化
create extension if not exists vector;

-- ② 思い出を保存するテーブルを作成
create table if not exists trip_memories (
  id uuid primary key default gen_random_uuid(),
  date text not null,
  location text,
  text_memo text,
  image_paths jsonb default '[]'::jsonb,
  audio_path text,
  video_path text,
  pdf_path text,
  timestamp text,
  embedding vector(768)
);

-- ③ AI類似度検索用の関数を作成（コサイン距離が近い順に取得）
create or replace function match_memories (
  query_embedding vector(768),
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
  from trip_memories
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- ④ メディア保存用のStorageバケットを作成し、公開アクセスを許可
insert into storage.buckets (id, name, public) values ('media', 'media', true) on conflict do nothing;
create policy "Public Access" on storage.objects for select using ( bucket_id = 'media' );
create policy "Public Insert" on storage.objects for insert with check ( bucket_id = 'media' );