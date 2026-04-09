import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

import ConversationPane from './components/demo/ConversationPane';
import EnginePane from './components/demo/EnginePane';
import SettingsPanel from './components/demo/SettingsPanel';
import { DEMO_CONFIG } from './demo/config';
import { useStreamingMemoryDemo } from './hooks/useStreamingMemoryDemo';

export default function Demo() {
  const {
    messages,
    input,
    setInput,
    isStreaming,
    showSettings,
    setShowSettings,
    waitingForFirstToken,
    updateFrequency,
    setUpdateFrequency,
    maxMemories,
    setMaxMemories,
    lookbackTokens,
    setLookbackTokens,
    generatedTokens,
    liveStreamingTokens,
    liveAllTokens,
    uniqueMemoriesCount,
    memoryUpdates,
    currentMemories,
    thinking,
    thinkingPrefix,
    revisions,
    activeRevisionId,
    contextSnapshot,
    inputRef,
    chatRef,
    lastMessageRef,
    handleScroll,
    sendMessage,
  } = useStreamingMemoryDemo();

  const hasMessages = messages.length > 0;

  return (
    <div className="h-screen overflow-hidden bg-white text-[#1a1a1a]">
      <div className="flex h-full flex-col">
        <div className="flex items-center justify-between border-b border-[#eee] px-6 py-3">
          <div className="flex items-center gap-6">
            <Link
              to="/"
              className="text-xs text-[#999] transition hover:text-[#666]"
            >
              Back
            </Link>
            <div className="h-4 w-px bg-[#eee]" />
            <span className="text-xs text-[#999]">
              {DEMO_CONFIG.name}
            </span>
          </div>

          <div className="relative">
            <button
              type="button"
              onClick={() => setShowSettings(!showSettings)}
              className="text-xs text-[#999] transition hover:text-[#666]"
            >
              Settings
            </button>
            <SettingsPanel
              open={showSettings}
              isStreaming={isStreaming}
              updateFrequency={updateFrequency}
              setUpdateFrequency={setUpdateFrequency}
              maxMemories={maxMemories}
              setMaxMemories={setMaxMemories}
              lookbackTokens={lookbackTokens}
              setLookbackTokens={setLookbackTokens}
            />
          </div>
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[minmax(560px,3fr)_minmax(340px,2fr)]">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="min-h-0 border-r border-[#eee] p-6"
          >
            <EnginePane
              contextSnapshot={contextSnapshot}
              activeRevisionId={activeRevisionId}
              revisions={revisions}
              generatedTokens={generatedTokens}
              liveStreamingTokens={liveStreamingTokens}
              liveAllTokens={liveAllTokens}
              currentMemories={currentMemories}
              uniqueMemoriesCount={uniqueMemoriesCount}
              memoryUpdates={memoryUpdates}
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.05 }}
            className="flex min-h-0 flex-col p-6"
          >
            <ConversationPane
              hasMessages={hasMessages}
              messages={messages}
              input={input}
              setInput={setInput}
              sendMessage={sendMessage}
              isStreaming={isStreaming}
              waitingForFirstToken={waitingForFirstToken}
              thinking={thinking}
              thinkingPrefix={thinkingPrefix}
              inputRef={inputRef}
              chatRef={chatRef}
              lastMessageRef={lastMessageRef}
              handleScroll={handleScroll}
              placeholder={DEMO_CONFIG.placeholder}
              headline={DEMO_CONFIG.headline}
              description={DEMO_CONFIG.description}
              suggestedQuestions={DEMO_CONFIG.suggestedQuestions}
            />
          </motion.div>
        </div>
      </div>
    </div>
  );
}
