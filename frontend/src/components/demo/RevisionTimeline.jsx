import { motion } from 'framer-motion';

export default function RevisionTimeline({
  revisions,
  activeRevisionId,
  generatedTokens,
}) {
  return (
    <div className="rounded-2xl border border-[#eee] bg-white p-5">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-[0.2em] text-[#999]">
            Revision timeline
          </div>
          <div className="mt-1 text-sm text-[#666]">
            A new pulse appears whenever the engine rewrites the prompt.
          </div>
        </div>
        <div className="font-mono text-xs text-[#999]">
          {generatedTokens.toLocaleString()} generated
        </div>
      </div>

      <div className="flex items-center gap-2 overflow-x-auto pb-1">
        {revisions.length === 0 && (
          <div className="text-xs text-[#999]">
            Waiting for the first context revision.
          </div>
        )}

        {revisions.map((revision) => {
          const isActive = revision.id === activeRevisionId;

          return (
            <div
              key={revision.id}
              className="flex min-w-fit items-center gap-2"
            >
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className={[
                  'flex h-9 w-9 items-center justify-center rounded-full border text-[11px] font-mono',
                  isActive
                    ? 'border-[#1a1a1a] bg-[#fafafa] text-[#1a1a1a]'
                    : 'border-[#eee] bg-white text-[#999]',
                ].join(' ')}
                title={
                  'Revision ' +
                  revision.id +
                  ' • ' +
                  revision.memoryCount +
                  ' memories • ' +
                  revision.generatedTokens +
                  ' generated'
                }
              >
                {revision.id}
              </motion.div>
              <div className="min-w-24">
                <div className="font-mono text-[11px] text-[#1a1a1a]">
                  {revision.tokenCount.toLocaleString()} tok
                </div>
                <div className="text-[11px] text-[#999]">
                  {revision.memoryCount} memories
                </div>
              </div>
              {revision !== revisions[revisions.length - 1] && (
                <div className="h-px w-10 bg-[#eee]" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
