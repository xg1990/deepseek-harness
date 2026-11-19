/**
 * Streamable HTTP session loss: a 404 on a request that carried `Mcp-Session-Id` starts a replacement
 * session. The real SDK client talks to a localhost fake that issues sessions and can forget them.
 */
import { once } from 'node:events'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { ToolCallId } from '@deepseek-ai/dsh-llm/types'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import * as NativeToolExecutor from '@deepseek-ai/dsh-tool-executor-native'
import { executeToolFixture } from '@deepseek-ai/dsh-tool-testkit'
import { Config as ConfigSchema } from '@deepseek-ai/dsh-mcp-client/src/index.ts'
import { resolveReconnectPolicy, startConnection } from '@deepseek-ai/dsh-mcp-client/src/connection.ts'
import type { ConnectionHandle, ReconnectConfig } from '@deepseek-ai/dsh-mcp-client/src/connection.ts'
import { McpStatusCell } from '@deepseek-ai/dsh-mcp-client/src/status.ts'

const FAST: ReconnectConfig = { initialDelayMs: 5, maxDelayMs: 20, maxAttempts: 3 }
const testToolSignal = new AbortController().signal
let callSeq = 0

interface Post {
  method: string
  sessionId: string | undefined
}

interface FixtureOptions {
  /** Status answered to `initialize` before any session exists; omit to issue sessions. */
  initializeStatus?: number
  /** Hold each GET stream until `releaseStream()`, then answer with this status; omit to answer 405 at once. */
  streamStatus?: number
  /** Issue no session ID and accept every request without one. */
  stateless?: boolean
}

interface SessionServer {
  url: string
  /** POST requests in arrival order, with the `Mcp-Session-Id` each carried. */
  posts: Post[]
  /** GET stream requests answered so far. */
  readonly streamsAnswered: number
  /** Forget every issued session, as a restarted or evicting server does. */
  forgetSessions(): void
  /** Answer the next `tools/call` with this status and the body `boom`. */
  failNextToolCall(status: number): void
  /** Answer the held GET stream request. */
  releaseStream(): void
  close(): Promise<void>
}

function sendJson(response: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  response.writeHead(status, { 'content-type': 'application/json', ...headers }).end(JSON.stringify(body))
}

function resultFor(method: string): unknown {
  if (method === 'tools/list') return { tools: [{ name: 'remote', inputSchema: { type: 'object' } }] }
  if (method === 'tools/call') return { content: [{ type: 'text', text: 'ok' }] }
  return {}
}

async function startSessionServer(options: FixtureOptions): Promise<SessionServer> {
  const sessions = new Set<string>()
  const posts: Post[] = []
  const streamGate: PromiseWithResolvers<void> = Promise.withResolvers()
  let issued = 0
  let streamsAnswered = 0
  let toolCallFailure: number | undefined

  async function answer(request: IncomingMessage, response: ServerResponse): Promise<void> {
    if (request.method === 'GET') {
      if (options.streamStatus === undefined) {
        response.writeHead(405).end()
      } else {
        await streamGate.promise
        response.writeHead(options.streamStatus).end()
      }
      streamsAnswered += 1
      return
    }
    if (request.method !== 'POST') {
      response.writeHead(405).end()
      return
    }
    const chunks: Buffer[] = []
    for await (const chunk of request) chunks.push(chunk as Buffer)
    const message = JSON.parse(Buffer.concat(chunks).toString('utf8')) as {
      id?: number
      method: string
      params?: { protocolVersion?: string }
    }
    const header = request.headers['mcp-session-id']
    const sessionId = typeof header === 'string' ? header : undefined
    posts.push({ method: message.method, sessionId })
    if (message.method === 'server/discover') {
      // Method not found makes the SDK's auto negotiation fall back to the legacy initialize.
      sendJson(response, 200, { jsonrpc: '2.0', id: message.id, error: { code: -32601, message: 'Method not found' } })
      return
    }
    if (message.method === 'initialize') {
      if (options.initializeStatus !== undefined) {
        response.writeHead(options.initializeStatus, { 'content-type': 'text/plain' }).end('not found\n')
        return
      }
      issued += 1
      const id = `session-${issued}`
      sessions.add(id)
      sendJson(response, 200, {
        jsonrpc: '2.0',
        id: message.id,
        result: { protocolVersion: message.params?.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: 'session-fixture', version: '1.0.0' } },
      }, options.stateless === true ? {} : { 'mcp-session-id': id })
      return
    }
    if (options.stateless !== true && (sessionId === undefined || !sessions.has(sessionId))) {
      response.writeHead(404, { 'content-type': 'text/plain' }).end('session not found\n')
      return
    }
    if (message.id === undefined) {
      response.writeHead(202).end()
      return
    }
    if (message.method === 'tools/call' && toolCallFailure !== undefined) {
      response.writeHead(toolCallFailure, { 'content-type': 'text/plain' }).end('boom')
      toolCallFailure = undefined
      return
    }
    sendJson(response, 200, { jsonrpc: '2.0', id: message.id, result: resultFor(message.method) })
  }

  const server: Server = createServer((request, response) => {
    answer(request, response).catch((error: unknown) => {
      response.writeHead(500).end(String(error))
    })
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('session fixture has no TCP address')
  return {
    url: `http://127.0.0.1:${address.port}/mcp`,
    posts,
    get streamsAnswered() { return streamsAnswered },
    forgetSessions: () => { sessions.clear() },
    failNextToolCall: (status) => { toolCallFailure = status },
    releaseStream: () => { streamGate.resolve() },
    close: () => new Promise<void>((resolve, reject) => {
      server.closeAllConnections()
      server.close((error) => { if (error === undefined) resolve(); else reject(error) })
    }),
  }
}

interface Harness {
  ctx: Context
  handle: ConnectionHandle
  server: SessionServer
  warns: string[]
  errors: string[]
  /** Successful connections reported by the status cell so far, including the first. */
  connections: () => number
  /** Status updates reported by the supervisor. */
  statuses: () => ReturnType<McpStatusCell['current']>[]
}

/** Connect one client to a fresh fake server; teardown runs even when an assertion fails. */
async function withSession(fixture: FixtureOptions, reconnect: ReconnectConfig, body: (harness: Harness) => Promise<void>): Promise<void> {
  const server = await startSessionServer(fixture)
  const ctx = new Context()
  const warns: string[] = []
  const errors: string[] = []
  try {
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(NativeToolExecutor)
    ctx.logger.warn = ((message: unknown) => { warns.push(String(message)) }) as typeof ctx.logger.warn
    ctx.logger.error = ((message: unknown) => { errors.push(String(message)) }) as typeof ctx.logger.error
    const config = ConfigSchema({
      transport: 'streamable-http', serverName: 'session', url: server.url,
      toolCallTimeoutMs: 60_000, startupTimeoutMs: 30_000, failOnStartupError: false,
    })
    const status = new McpStatusCell(() => {})
    let connections = 0
    const statuses: ReturnType<McpStatusCell['current']>[] = []
    status.onChange((value) => {
      statuses.push(value)
      if (value.state === 'connected') connections += 1
    })
    const handle = startConnection(ctx, config, resolveReconnectPolicy(reconnect, 'session'), status)
    try {
      await handle.ready
      await body({ ctx, handle, server, warns, errors, connections: () => connections, statuses: () => statuses })
    } finally {
      await handle.dispose()
    }
  } finally {
    await ctx.fiber.dispose()
    await server.close()
  }
}

function callRemote(ctx: Context) {
  callSeq += 1
  return executeToolFixture(ctx, {
    name: 'mcp__session__remote', arguments: {}, callId: brandString<ToolCallId>(`session-call-${callSeq}`), signal: testToolSignal,
  })
}

function initializes(server: SessionServer): number {
  return server.posts.filter(post => post.method === 'initialize').length
}

/** Wait until the status cell reports `count` connections. A report follows tool sync, so calls can succeed. */
async function untilConnections(connections: () => number, count: number): Promise<void> {
  await vi.waitFor(() => { expect(connections()).toBe(count) }, { timeout: 3_000 })
}

function sleep(ms: number): Promise<void> {
  // Annotated binding (not withResolvers<void>()): reconnect.spec.ts explains the lint constraint.
  const gate: PromiseWithResolvers<void> = Promise.withResolvers()
  setTimeout(gate.resolve, ms)
  return gate.promise
}

describe('Streamable HTTP session loss', () => {
  it('serves tool calls on the first session', async () => {
    await withSession({}, FAST, async ({ ctx, server }) => {
      expect((await callRemote(ctx)).isError).toBe(false)
      expect(server.posts.at(-1)).toEqual({ method: 'tools/call', sessionId: 'session-1' })
    })
  })

  it('opens a replacement session after the server forgets the session', async () => {
    await withSession({}, FAST, async ({ ctx, server, warns, connections }) => {
      expect((await callRemote(ctx)).isError).toBe(false)
      server.forgetSessions()
      const failed = await callRemote(ctx)
      expect(failed.error?.message).toContain('session not found')
      await untilConnections(connections, 2)
      expect((await callRemote(ctx)).isError).toBe(false)
      expect(server.posts.filter(post => post.method === 'initialize')).toEqual([
        { method: 'initialize', sessionId: undefined },
        { method: 'initialize', sessionId: undefined },
      ])
      expect(server.posts.at(-1)).toEqual({ method: 'tools/call', sessionId: 'session-2' })
      expect(warns.filter(line => line.includes('connection lost; reconnecting'))).toHaveLength(1)
    })
  })

  it('marks the connection failed and does not reconnect when reconnect is disabled', async () => {
    const disabled: ReconnectConfig = { enabled: false, initialDelayMs: 1, maxDelayMs: 1, maxAttempts: 3 }
    await withSession({}, disabled, async ({ ctx, server, errors, statuses }) => {
      expect((await callRemote(ctx)).isError).toBe(false)
      server.forgetSessions()
      const failed = await callRemote(ctx)
      expect(failed.error?.message).toContain('session not found')
      await vi.waitFor(() => {
        expect(statuses().some(value => value.state === 'failed' && value.reason === 'reconnect-disabled')).toBe(true)
      }, { timeout: 3_000 })
      expect(errors.some(line => line.includes('reconnect is disabled'))).toBe(true)
      await sleep(50)
      expect(initializes(server)).toBe(1)
      expect((await callRemote(ctx)).isError).toBe(true)
    })
  })

  it('treats a 404 before any session as a failed connection attempt', async () => {
    const retries: ReconnectConfig = { initialDelayMs: 1, maxDelayMs: 1, maxAttempts: 2 }
    await withSession({ initializeStatus: 404 }, retries, async ({ handle, warns, errors }) => {
      expect((await handle.ready).error).toBeDefined()
      await vi.waitFor(() => { expect(errors.some(line => line.includes('giving up after 2'))).toBe(true) }, { timeout: 3_000 })
      expect(warns.some(line => line.includes('connection lost'))).toBe(false)
      expect(warns.filter(line => line.includes('connection failed; retrying'))).toHaveLength(2)
    })
  })

  it('keeps the session when a request fails with a status other than 404', async () => {
    await withSession({}, FAST, async ({ ctx, server, warns }) => {
      server.failNextToolCall(500)
      expect((await callRemote(ctx)).error?.message).toContain('boom')
      expect((await callRemote(ctx)).isError).toBe(false)
      expect(server.posts.at(-1)).toEqual({ method: 'tools/call', sessionId: 'session-1' })
      expect(initializes(server)).toBe(1)
      expect(warns.some(line => line.includes('connection lost'))).toBe(false)
    })
  })

  it('does not reconnect on a 404 from a server that issues no session', async () => {
    await withSession({ stateless: true }, FAST, async ({ ctx, server, warns }) => {
      expect((await callRemote(ctx)).isError).toBe(false)
      server.failNextToolCall(404)
      expect((await callRemote(ctx)).error?.message).toContain('boom')
      expect((await callRemote(ctx)).isError).toBe(false)
      expect(initializes(server)).toBe(1)
      expect(warns.some(line => line.includes('connection lost'))).toBe(false)
    })
  })

  it('starts one replacement session when concurrent calls hit the lost session', async () => {
    await withSession({}, FAST, async ({ ctx, server, warns, connections }) => {
      expect((await callRemote(ctx)).isError).toBe(false)
      server.forgetSessions()
      // The first 404 keeps its own message; a sibling still in flight when the generation closes reports a closed connection.
      const failures = await Promise.all([callRemote(ctx), callRemote(ctx)])
      expect(failures.map(failure => failure.isError)).toEqual([true, true])
      expect(failures.some(failure => failure.error?.message.includes('session not found'))).toBe(true)
      await untilConnections(connections, 2)
      expect(initializes(server)).toBe(2)
      expect(warns.filter(line => line.includes('connection lost; reconnecting'))).toHaveLength(1)
      expect((await callRemote(ctx)).isError).toBe(false)
    })
  })

  it('does not reconnect on a 404 from the GET stream', async () => {
    await withSession({ streamStatus: 404 }, FAST, async ({ ctx, server, warns }) => {
      expect((await callRemote(ctx)).isError).toBe(false)
      server.releaseStream()
      await vi.waitFor(() => { expect(server.streamsAnswered).toBe(1) }, { timeout: 3_000 })
      await sleep(50)
      expect((await callRemote(ctx)).isError).toBe(false)
      expect(initializes(server)).toBe(1)
      expect(warns.some(line => line.includes('connection lost'))).toBe(false)
    })
  })
})
