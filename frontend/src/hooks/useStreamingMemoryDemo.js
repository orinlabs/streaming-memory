import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import { API_URL as DEFAULT_API_URL } from '../demo/config';
import { buildContextSnapshot } from '../demo/context';

function updateAssistantMessage(messages, assistantId, updates) {
  return messages.map((message) => {
    if (message.id !== assistantId) {
      return message;
    }

    return {
      ...message,
      ...updates,
    };
  });
}

export function useStreamingMemoryDemo(apiUrlOverride) {
  const API_URL = apiUrlOverride || DEFAULT_API_URL;
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [waitingForFirstToken, setWaitingForFirstToken] = useState(false);

  const [updateFrequency, setUpdateFrequency] = useState(10);
  const [maxMemories, setMaxMemories] = useState(8);
  const [lookbackTokens, setLookbackTokens] = useState(60);

  const [baseContextSize, setBaseContextSize] = useState(0);
  const [currentMemoryTokens, setCurrentMemoryTokens] = useState(0);
  const [ragMemoryTokens, setRagMemoryTokens] = useState(0);
  const [allMemoriesTokens, setAllMemoriesTokens] = useState(0);
  const [uniqueMemoriesCount, setUniqueMemoriesCount] = useState(0);
  const [generatedTokens, setGeneratedTokens] = useState(0);
  const [liveStreamingTokens, setLiveStreamingTokens] = useState(0);
  const [liveRagTokens, setLiveRagTokens] = useState(0);
  const [liveAllTokens, setLiveAllTokens] = useState(0);
  const [memoryUpdates, setMemoryUpdates] = useState(0);

  const [currentMemories, setCurrentMemories] = useState([]);
  const [thinkingPrefix, setThinkingPrefix] = useState('');
  const [thinking, setThinking] = useState('');
  const [response, setResponse] = useState('');

  const [requestHistory, setRequestHistory] = useState([]);
  const [currentUserMessage, setCurrentUserMessage] = useState('');
  const [revisions, setRevisions] = useState([]);
  const [activeRevisionId, setActiveRevisionId] = useState(0);

  const [userScrolledUp, setUserScrolledUp] = useState(false);

  const inputRef = useRef(null);
  const chatRef = useRef(null);
  const lastMessageRef = useRef(null);
  const revisionCounterRef = useRef(0);

  useEffect(() => {
    fetch(API_URL + '/health')
      .then((res) => {
        if (!res.ok) {
          console.error('Health check failed:', res.status, res.statusText);
        }
      })
      .catch((err) => {
        console.error('Failed to connect to API:', err.message);
      });
  }, []);

  useEffect(() => {
    if (isStreaming) {
      setUserScrolledUp(false);
    }
  }, [isStreaming]);

  useEffect(() => {
    if (isStreaming && chatRef.current && !userScrolledUp) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [isStreaming, messages, thinking, response, userScrolledUp]);

  const contextSnapshot = useMemo(() => {
    return buildContextSnapshot({
      history: requestHistory,
      userMessage: currentUserMessage,
      currentMemories,
      thinking,
      response,
    });
  }, [
    currentMemories,
    currentUserMessage,
    requestHistory,
    response,
    thinking,
  ]);

  const scrollToNewMessage = () => {
    if (!lastMessageRef.current || !chatRef.current) {
      return;
    }

    const containerTop = chatRef.current.getBoundingClientRect().top;
    const messageTop = lastMessageRef.current.getBoundingClientRect().top;
    const offset = messageTop - containerTop - 40;

    chatRef.current.scrollTop += offset;
  };

  const handleScroll = () => {
    if (!chatRef.current) {
      return;
    }

    const {
      scrollTop,
      scrollHeight,
      clientHeight,
    } = chatRef.current;

    const nearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setUserScrolledUp(!nearBottom);
  };

  const pushRevision = ({
    history,
    userMessage,
    memories,
    fullThinking,
    fullResponse,
    generatedTokenCount,
    streamingTokenCount,
  }) => {
    revisionCounterRef.current += 1;

    const snapshot = buildContextSnapshot({
      history,
      userMessage,
      currentMemories: memories,
      thinking: fullThinking,
      response: fullResponse,
    });

    const revision = {
      id: revisionCounterRef.current,
      generatedTokens: generatedTokenCount,
      memoryCount: memories.length,
      tokenCount: streamingTokenCount,
      lines: snapshot.lines,
      text: snapshot.text,
      createdAt: Date.now(),
    };

    setRevisions((prev) => prev.concat([revision]));
    setActiveRevisionId(revision.id);
  };

  const resetRunState = () => {
    setThinkingPrefix('');
    setThinking('');
    setResponse('');
    setCurrentMemories([]);
    setBaseContextSize(0);
    setCurrentMemoryTokens(0);
    setRagMemoryTokens(0);
    setAllMemoriesTokens(0);
    setUniqueMemoriesCount(0);
    setGeneratedTokens(0);
    setLiveStreamingTokens(0);
    setLiveRagTokens(0);
    setLiveAllTokens(0);
    setMemoryUpdates(0);
    setRevisions([]);
    setActiveRevisionId(0);
    revisionCounterRef.current = 0;
  };

  const sendMessage = async (directMessage) => {
    const messageToSend = directMessage || input.trim();

    if (!messageToSend || isStreaming) {
      return;
    }

    const userMessage = messageToSend;
    const assistantId = Date.now();
    const history = messages.map((message) => ({
      role: message.role,
      content: message.content,
    }));

    setInput('');
    setIsStreaming(true);
    setWaitingForFirstToken(true);
    resetRunState();
    setRequestHistory(history);
    setCurrentUserMessage(userMessage);

    setMessages((prev) => prev.concat([
      { role: 'user', content: userMessage },
      {
        role: 'assistant',
        content: '',
        id: assistantId,
        streaming: true,
      },
    ]));

    window.setTimeout(scrollToNewMessage, 50);

    try {
      const streamResponse = await fetch(API_URL + '/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          history,
          update_every_n: updateFrequency,
          max_memories: maxMemories,
          lookback_tokens: lookbackTokens,
          scenario: 'dad',
        }),
      });

      if (!streamResponse.ok || !streamResponse.body) {
        throw new Error('Stream request failed with status ' + streamResponse.status);
      }

      const reader = streamResponse.body.getReader();
      const decoder = new TextDecoder();

      let buffer = '';
      let fullResponse = '';
      let fullThinking = '';
      let latestMemories = [];
      let latestGeneratedTokens = 0;
      let latestStreamingTokenCount = 0;
      let hasInitialRevision = false;

      while (true) {
        const chunk = await reader.read();

        if (chunk.done) {
          break;
        }

        buffer += decoder.decode(chunk.value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const event of events) {
          if (!event.trim()) {
            continue;
          }

          const match = event.match(/^data:\s*(.+)$/s);

          if (!match) {
            continue;
          }

          try {
            const data = JSON.parse(match[1]);

            if (data.type === 'context_size') {
              if (data.base_context_size !== undefined) {
                setBaseContextSize(data.base_context_size);
              }
              if (data.current_memory_tokens !== undefined) {
                setCurrentMemoryTokens(data.current_memory_tokens);
              }
              if (data.rag_memory_tokens !== undefined) {
                setRagMemoryTokens(data.rag_memory_tokens);
              }
              if (data.all_memories_tokens !== undefined) {
                setAllMemoriesTokens(data.all_memories_tokens);
              }
              if (data.unique_memories !== undefined) {
                setUniqueMemoriesCount(data.unique_memories);
              }

              latestStreamingTokenCount =
                (data.base_context_size || 0) + (data.current_memory_tokens || 0);

              setLiveStreamingTokens(latestStreamingTokenCount);
              setLiveRagTokens(
                (data.base_context_size || 0) + (data.rag_memory_tokens || 0)
              );
              setLiveAllTokens(
                (data.base_context_size || 0) + (data.all_memories_tokens || 0)
              );

              if (!hasInitialRevision && latestMemories.length > 0) {
                hasInitialRevision = true;
                pushRevision({
                  history,
                  userMessage,
                  memories: latestMemories,
                  fullThinking,
                  fullResponse,
                  generatedTokenCount: latestGeneratedTokens,
                  streamingTokenCount: latestStreamingTokenCount,
                });
              }
            } else if (data.type === 'memories') {
              latestMemories = data.memories || [];
              setCurrentMemories(latestMemories);
              setWaitingForFirstToken(false);
            } else if (data.type === 'thinking_prefix') {
              setThinkingPrefix(data.t || '');
            } else if (data.type === 'memory_update') {
              latestMemories = data.memories || [];
              setCurrentMemories(latestMemories);
              setMemoryUpdates((prev) => prev + 1);

              if (data.base_context_size !== undefined) {
                setBaseContextSize(data.base_context_size);
              }
              if (data.current_memory_tokens !== undefined) {
                setCurrentMemoryTokens(data.current_memory_tokens);
              }
              if (data.rag_memory_tokens !== undefined) {
                setRagMemoryTokens(data.rag_memory_tokens);
              }
              if (data.all_memories_tokens !== undefined) {
                setAllMemoriesTokens(data.all_memories_tokens);
              }
              if (data.unique_memories !== undefined) {
                setUniqueMemoriesCount(data.unique_memories);
              }

              latestStreamingTokenCount =
                (data.base_context_size || 0) + (data.current_memory_tokens || 0);

              setLiveStreamingTokens(latestStreamingTokenCount);
              setLiveRagTokens(
                (data.base_context_size || 0) + (data.rag_memory_tokens || 0)
              );
              setLiveAllTokens(
                (data.base_context_size || 0) + (data.all_memories_tokens || 0)
              );

              pushRevision({
                history,
                userMessage,
                memories: latestMemories,
                fullThinking,
                fullResponse,
                generatedTokenCount: latestGeneratedTokens,
                streamingTokenCount: latestStreamingTokenCount,
              });
            } else if (data.type === 'context_update') {
              latestGeneratedTokens = data.generated_tokens || 0;
              setGeneratedTokens(latestGeneratedTokens);

              if (data.streaming !== undefined) {
                latestStreamingTokenCount = data.streaming;
                setLiveStreamingTokens(data.streaming);
              }
              if (data.rag !== undefined) {
                setLiveRagTokens(data.rag);
              }
              if (data.all !== undefined) {
                setLiveAllTokens(data.all);
              }
            } else if (data.type === 'thinking') {
              fullThinking += data.t || '';
              setThinking(fullThinking);
              setMessages((prev) =>
                updateAssistantMessage(prev, assistantId, {
                  thinking: fullThinking,
                })
              );
            } else if (data.type === 'token') {
              fullResponse += data.t || '';
              setResponse(fullResponse);
              setMessages((prev) =>
                updateAssistantMessage(prev, assistantId, {
                  content: fullResponse,
                })
              );
            } else if (data.type === 'max_tokens') {
              setMessages((prev) =>
                updateAssistantMessage(prev, assistantId, {
                  hitMaxTokens: true,
                })
              );
            } else if (data.type === 'done') {
              setMessages((prev) =>
                updateAssistantMessage(prev, assistantId, {
                  streaming: false,
                  content: fullResponse,
                  thinking: fullThinking,
                })
              );
            }
          } catch (error) {
            console.error('Parse error:', error);
          }
        }
      }
    } catch (error) {
      console.error('Chat stream error:', error);

      let errorMessage = error.message;

      if (error.message && error.message.includes('fetch')) {
        errorMessage = [
          'Failed to connect to API. Please check:',
          '1. Modal service is deployed and running',
          '2. API URL is correct: ' + API_URL,
          '3. No network/firewall blocking the connection',
          '',
          'Error: ' + error.message,
        ].join('\n');
      }

      setMessages((prev) =>
        updateAssistantMessage(prev, assistantId, {
          content: errorMessage,
          streaming: false,
          error: true,
        })
      );
    }

    setIsStreaming(false);
    setWaitingForFirstToken(false);
    inputRef.current?.focus();
  };

  return {
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
    liveRagTokens,
    liveAllTokens,
    baseContextSize,
    currentMemoryTokens,
    ragMemoryTokens,
    allMemoriesTokens,
    uniqueMemoriesCount,
    memoryUpdates,
    currentMemories,
    thinkingPrefix,
    thinking,
    response,
    revisions,
    activeRevisionId,
    contextSnapshot,
    requestHistory,
    currentUserMessage,
    inputRef,
    chatRef,
    lastMessageRef,
    handleScroll,
    sendMessage,
  };
}
