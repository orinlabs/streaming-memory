import TerminalContextPanel from './TerminalContextPanel';

export default function EnginePane({
  contextSnapshot,
  activeRevisionId,
  revisions,
  generatedTokens,
  liveStreamingTokens,
  liveAllTokens,
  currentMemories,
  uniqueMemoriesCount,
  memoryUpdates,
}) {
  return (
    <TerminalContextPanel
      lines={contextSnapshot.lines}
      activeRevisionId={activeRevisionId}
      revisionCount={revisions.length}
      generatedTokens={generatedTokens}
      contextTokenCount={liveStreamingTokens}
      currentMemories={currentMemories}
      uniqueMemoriesCount={uniqueMemoriesCount}
      memoryUpdates={memoryUpdates}
      allMemoriesTokens={liveAllTokens}
    />
  );
}
