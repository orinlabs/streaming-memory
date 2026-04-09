function FootprintBar({
  label,
  value,
  maxValue,
  colorClass,
  valueClass,
}) {
  const width = maxValue > 0 ? Math.max((value / maxValue) * 100, 4) : 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-4 text-xs">
        <span className="text-[#666]">{label}</span>
        <span className={['font-mono', valueClass].join(' ')}>
          {value.toLocaleString()} tokens
        </span>
      </div>
      <div className="h-2 rounded-full bg-[#f2f2f2]">
        <div
          className={['h-2 rounded-full transition-all duration-300', colorClass].join(' ')}
          style={{ width: width + '%' }}
        />
      </div>
    </div>
  );
}

export default function TokenFootprint({
  generatedTokens,
  uniqueMemoriesCount,
  liveStreamingTokens,
  liveRagTokens,
  liveAllTokens,
}) {
  const maxValue = Math.max(liveStreamingTokens, liveRagTokens, liveAllTokens, 1);

  return (
    <div className="rounded-2xl border border-[#eee] bg-white p-5">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-[0.2em] text-[#999]">
            Token footprint
          </div>
          <div className="mt-1 text-sm text-[#666]">
            Streaming memory keeps the active prompt compact.
          </div>
        </div>
        <div className="text-right text-xs text-[#999]">
          <div>{uniqueMemoriesCount} memories touched</div>
          <div>{generatedTokens} tokens generated</div>
        </div>
      </div>

      <div className="space-y-4">
        <FootprintBar
          label="Prompt stuffing"
          value={liveAllTokens}
          maxValue={maxValue}
          colorClass="bg-red-400"
          valueClass="text-red-400"
        />
        <FootprintBar
          label="One-shot RAG"
          value={liveRagTokens}
          maxValue={maxValue}
          colorClass="bg-orange-400"
          valueClass="text-orange-400"
        />
        <FootprintBar
          label="Streaming memory"
          value={liveStreamingTokens}
          maxValue={maxValue}
          colorClass="bg-green-500"
          valueClass="text-green-600"
        />
      </div>
    </div>
  );
}
