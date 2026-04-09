import { motion } from 'framer-motion';

function Composer({ input, setInput, onSend, isStreaming, inputRef, placeholder }) {
  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="text"
        value={input}
        disabled={isStreaming}
        placeholder={placeholder}
        onChange={(event) => setInput(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            onSend();
          }
        }}
        className="w-full rounded-full bg-[#f5f5f5] px-4 py-3 pr-20 text-sm text-[#1a1a1a] outline-none placeholder:text-[#999] focus:ring-2 focus:ring-[#ddd] disabled:bg-[#eee]"
      />
      <button
        type="button"
        disabled={isStreaming || !input.trim()}
        onClick={() => onSend()}
        className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-[#1a1a1a] px-4 py-1.5 text-xs font-medium text-white transition hover:bg-[#333] disabled:bg-[#ddd] disabled:text-[#999]"
      >
        {isStreaming ? 'Streaming...' : 'Send'}
      </button>
    </div>
  );
}

function EmptyState({ headline, description, suggestedQuestions, onSelectQuestion }) {
  return (
    <div className="flex h-full flex-col justify-center">
      <h1 className="max-w-md text-2xl font-bold text-[#1a1a1a]">
        {headline}
      </h1>
      <p className="mt-3 max-w-md text-sm leading-6 text-[#666]">
        {description}
      </p>
      <div className="mt-6 flex flex-wrap gap-2">
        {suggestedQuestions.map((question) => (
          <button
            key={question}
            type="button"
            onClick={() => onSelectQuestion(question)}
            className="rounded-full bg-[#f5f5f5] px-3 py-1.5 text-sm text-[#666] transition hover:bg-[#eee]"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ConversationPane({
  hasMessages,
  messages,
  input,
  setInput,
  sendMessage,
  isStreaming,
  waitingForFirstToken,
  thinking,
  thinkingPrefix,
  inputRef,
  chatRef,
  lastMessageRef,
  handleScroll,
  placeholder,
  headline,
  description,
  suggestedQuestions,
}) {
  const latestUserMessage = [...messages]
    .reverse()
    .find((m) => m.role === 'user');
  const latestAssistantMessage = [...messages]
    .reverse()
    .find((m) => m.role === 'assistant');

  return (
    <div className="flex h-full flex-col">
      {!hasMessages ? (
        <div className="flex flex-1 flex-col justify-center">
          <EmptyState
            headline={headline}
            description={description}
            suggestedQuestions={suggestedQuestions}
            onSelectQuestion={sendMessage}
          />
        </div>
      ) : (
        <div
          ref={chatRef}
          onScroll={handleScroll}
          className="flex min-h-0 flex-1 flex-col gap-4"
        >
          {latestUserMessage && (
            <motion.div
              key={latestUserMessage.content}
              ref={lastMessageRef}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="text-xs text-[#999] mb-1">You asked</div>
              <p className="text-sm text-[#1a1a1a]">{latestUserMessage.content}</p>
            </motion.div>
          )}

          {isStreaming && waitingForFirstToken ? (
            <p className="text-sm text-[#999] italic">
              Warming up...
            </p>
          ) : (
            latestAssistantMessage && (
              <motion.div
                key={latestAssistantMessage.id || 'assistant'}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex min-h-0 flex-1 flex-col"
              >
                <div className="text-xs text-[#999] mb-1">Response</div>
                <div className="min-h-0 flex-1 overflow-y-auto text-sm leading-7">
                  {latestAssistantMessage.streaming && thinking && (
                    <div className="mb-3 whitespace-pre-wrap text-[#aaa] italic">
                      {thinking}
                    </div>
                  )}
                  {!latestAssistantMessage.streaming && latestAssistantMessage.thinking && (
                    <div className="mb-3 whitespace-pre-wrap text-[#aaa] italic">
                      {latestAssistantMessage.thinking}
                    </div>
                  )}
                  <div className="whitespace-pre-wrap text-[#1a1a1a]">
                    {latestAssistantMessage.content}
                    {latestAssistantMessage.streaming && <span className="cursor" />}
                  </div>
                </div>
              </motion.div>
            )
          )}
        </div>
      )}

      <div className="pt-4">
        <Composer
          input={input}
          setInput={setInput}
          onSend={sendMessage}
          isStreaming={isStreaming}
          inputRef={inputRef}
          placeholder={placeholder}
        />
      </div>
    </div>
  );
}
