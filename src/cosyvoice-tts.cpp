#include "cosyvoice-internal.h"
#include "cosyvoice-text-chunk.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

constexpr uint32_t COSYVOICE_TTS_FLAG_MASK = COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION | COSYVOICE_TTS_FLAG_SPLIT_TEXT | COSYVOICE_TTS_FLAG_FAST_SPLIT | COSYVOICE_TTS_FLAG_FADE_IN;
constexpr double COSYVOICE_TTS_FADE_IN_SECONDS = 0.02;

struct cosyvoice_tts_context : cosyvoice_tokenization_result_impl, cosyvoice_prompt, std::string
{
    cosyvoice_tts_context(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt)
        : cosyvoice_prompt(*prompt), ctx(ctx),
        flags(
#ifndef COSYVOICE_NO_ICU
            COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION |
#endif
            COSYVOICE_TTS_FLAG_SPLIT_TEXT | COSYVOICE_TTS_FLAG_FAST_SPLIT | COSYVOICE_TTS_FLAG_FADE_IN
        ),
        fade_samples(static_cast<uint32_t>(std::ceil(static_cast<double>(ctx->get_sample_rate()) * COSYVOICE_TTS_FADE_IN_SECONDS)))
    {
        const auto instruction_prefix = ctx->get_instruction_prefix();
        if (instruction_prefix)
        {
            instruction_cache = instruction_prefix;
            prefix_len = instruction_cache.size();
        }
        else prefix_len = 0;
    }

    bool tts_job(const char* text, const char* instruction, float speed, cosyvoice_inference_mode mode, cosyvoice_generated_speech_ptr result, cosyvoice_tts_audio_callback_t callback, void* user_data)
    {
        instruction_cache.resize(prefix_len);
        if (instruction)
        {
            instruction_cache.push_back(' ');
            instruction_cache.append(instruction);
        }
        cosyvoice_prompt_set(
            ctx,
            this,
            mode,
            instruction_cache.c_str());

        bool normalized = false;
        if (flags & COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION)
            normalized = cosyvoice_frontend_util_text_normalize(*this, text, static_cast<uint32_t>(strlen(text)), nullptr);
        const char* effective_text = normalized ? c_str() : text;

        // When splitting is disabled, do a single tokenize+synthesize pass.
        if (!(flags & COSYVOICE_TTS_FLAG_SPLIT_TEXT))
        {
            ctx->tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }

        const auto fragments = cosyvoice_internal::split_into_fragments(effective_text);
        if (fragments.size() <= 1)
        {
            ctx->tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }

        // Compute the maximum text-token budget per chunk:
        //   m = n_max_seq                                    (the LLM context window)
        //   o = SOS(1) + prompt_text + (task_token(1) + prompt_speech_tokens when present)
        //   r = max_token_text_ratio                         (decode-stage growth factor)
        // We need m > o + l * (1 + r) so the prefill plus decoded speech tokens fit; solving
        // for l yields the cap below. Falling back to a single call when the budget is
        // pathological keeps behavior consistent with the pre-chunking path.
        cosyvoice_context_params_t params;
        ctx->get_context_params(&params);
        cosyvoice_generation_config_t gen_config;
        ctx->get_generation_config(&gen_config);

        const uint32_t prompt_text_len = static_cast<uint32_t>(prompt_text.size());
        const uint32_t prompt_speech_len = llm_prompt_speech_tokens.second;
        const uint64_t o = 1ull + prompt_text_len
            + (prompt_speech_len != 0 ? 1ull + prompt_speech_len : 0ull);
        const float r = gen_config.max_token_text_ratio;

        if (params.n_max_seq <= o + 1u || !(r > 0.0f))
        {
            ctx->tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }
        const auto max_text_tokens = static_cast<std::size_t>(
            static_cast<float>(params.n_max_seq - o - 1u) / (1.0f + r));

        if (max_text_tokens == 0)
        {
            ctx->tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }

        // Fast-split path: tokenize each fragment once and merge token-ID vectors,
        // avoiding re-tokenization of each assembled chunk.
        if (flags & COSYVOICE_TTS_FLAG_FAST_SPLIT)
        {
            std::vector<std::vector<int>> fragment_tokens;
            fragment_tokens.reserve(fragments.size());
            {
                cosyvoice_tokenization_result_impl tok;
                for (const auto& fragment : fragments)
                {
                    tok.tokens.clear();
                    ctx->tokenize(fragment.c_str(), &tok, true);
                    fragment_tokens.push_back(tok.tokens);
                }
            }
            auto chunk_token_list = cosyvoice_internal::reassemble_by_token_budget(
                fragment_tokens, max_text_tokens);

            if (chunk_token_list.size() <= 1)
            {
                const auto& t = chunk_token_list.empty() ? fragment_tokens[0] : chunk_token_list[0];
                tokens = t;
                return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                    : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
            }

            if (result)
            {
                combined_pcm.clear();
                for (auto& chunk_tokens : chunk_token_list)
                {
                    if (ctx->is_stop_requested())
                    {
                        result->data = nullptr;
                        result->length = 0;
                        return false;
                    }
                    tokens = std::move(chunk_tokens);
                    cosyvoice_generated_speech part = {};
                    if (!cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, &part)
                        || !part.data || part.length == 0)
                    {
                        result->data = nullptr;
                        result->length = 0;
                        return false;
                    }
                    combined_pcm.insert(combined_pcm.end(), part.data, part.data + part.length);
                }
                result->data = combined_pcm.data();
                result->length = static_cast<uint32_t>(combined_pcm.size());
                return true;
            }

            for (auto& chunk_tokens : chunk_token_list)
            {
                if (ctx->is_stop_requested())
                    return false;
                tokens = std::move(chunk_tokens);
                if (!cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data))
                    return false;
            }
            return true;
        }

        // Slow-split path: tokenize each fragment only for counting,
        // reassemble text chunks, then re-tokenize each chunk.
        cosyvoice_tokenization_result_impl tokens_scratch;
        const auto token_count = [&](std::string_view fragment) -> std::size_t
            {
                tokens_scratch.tokens.clear();
                ctx->tokenize(fragment.data(), static_cast<uint32_t>(fragment.size()), &tokens_scratch, true);
                return tokens_scratch.get_n_tokens();
            };
        const auto chunks = cosyvoice_internal::reassemble_by_token_budget(
            fragments, max_text_tokens, token_count);

        if (chunks.size() <= 1)
        {
            ctx->tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }

        // Multi-chunk path: synthesize each chunk and copy its PCM into a context-owned buffer.
        // cosyvoice_tts() points result->data at an internal token2wav buffer that is overwritten
        // on the next call, so the copy must happen before the following synthesis begins.
        combined_pcm.clear();
        if (result)
        {
            for (const auto& chunk : chunks)
            {
                if (ctx->is_stop_requested())
                {
                    result->data = nullptr;
                    result->length = 0;
                    return false;
                }
                ctx->tokenize(chunk.c_str(), this, true);
                cosyvoice_generated_speech part = {};
                if (!cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, &part)
                    || !part.data || part.length == 0)
                {
                    result->data = nullptr;
                    result->length = 0;
                    return false;
                }
                combined_pcm.insert(combined_pcm.end(), part.data, part.data + part.length);
            }
            result->data = combined_pcm.data();
            result->length = static_cast<uint32_t>(combined_pcm.size());
        }
        else
            for (const auto& chunk : chunks)
            {
                if (ctx->is_stop_requested())
                    return false;
                ctx->tokenize(chunk.c_str(), this, true);
                if (!cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data))
                    return false;
            }
        return true;
    }

    bool cosyvoice_tts_stream_with_postprocess(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_tts_audio_callback_t callback, void* user_data)
    {
        bool ok;
        if (flags & COSYVOICE_TTS_FLAG_FADE_IN)
            ok = cosyvoice_tts_stream(ctx, token_ids, n_tokens, speed, this, callback, user_data);
        else
        {
            struct callback_wrapper_context
            {
                cosyvoice_tts_audio_callback_t callback;
                void* user_data;
                uint32_t fade_samples;
                uint32_t processed_samples = 0;
            } cb_ctx{ callback, user_data, fade_samples };

            auto callback_wrapper = [](const float* data, uint32_t length, void* user_data) -> bool
            {
                auto ctx = static_cast<callback_wrapper_context*>(user_data);
                if (ctx->processed_samples < ctx->fade_samples)
                {
                    const uint32_t fade_len = std::min<uint32_t>(ctx->fade_samples - ctx->processed_samples, length);
                    for (uint32_t i = 0; i < fade_len; ++i)
                    {
                        const float ramp = static_cast<float>(ctx->processed_samples + i + 1) / static_cast<float>(ctx->fade_samples + 1);
                        const_cast<float*>(data)[i] *= ramp;
                    }
                    ctx->processed_samples += fade_len;
                }
                return ctx->callback(data, length, ctx->user_data);
            };
            ok = cosyvoice_tts_stream(ctx, token_ids, n_tokens, speed, this, callback_wrapper, &cb_ctx);
        }

        return ok;
    }

    bool cosyvoice_tts_with_postprocess(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_generated_speech_ptr result)
    {
        if (!cosyvoice_tts(ctx, token_ids, n_tokens, speed, this, result))
            return false;
        if ((flags & COSYVOICE_TTS_FLAG_FADE_IN) && result->data && result->length != 0)
        {
            const uint32_t fade_len = std::min<uint32_t>(fade_samples, result->length);
            for (uint32_t i = 0; i < fade_len; ++i)
            {
                const float ramp = static_cast<float>(i + 1) / static_cast<float>(fade_len + 1);
                result->data[i] *= ramp;
            }
        }
        return true;
    }

    cosyvoice_context_t ctx;
    std::string instruction_cache;
    size_t prefix_len;
    uint32_t flags;
    uint32_t fade_samples;
    std::vector<float> combined_pcm;
};

cosyvoice_tts_context_t cosyvoice_tts_context_new(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt)
{
    return new cosyvoice_tts_context(ctx, prompt);
}

void cosyvoice_tts_context_free(cosyvoice_tts_context_t ctx)
{
    delete ctx;
}

void cosyvoice_tts_context_set_prompt(cosyvoice_tts_context_t ctx, cosyvoice_prompt_t prompt)
{
    *static_cast<cosyvoice_prompt_t>(ctx) = *prompt;
}

static void set_flag(cosyvoice_tts_context_t ctx, uint32_t flag, bool enabled)
{
    if (enabled)
        ctx->flags |= flag;
    else
        ctx->flags &= ~flag;
}

#define COSYVOICE_TTS_CONTEXT_SET_FLAG_IMPL(name, flag) \
    bool cosyvoice_tts_context_set_##name##_enabled(cosyvoice_tts_context_t ctx, bool enabled) \
    { \
        set_flag(ctx, flag, enabled); \
        return true; \
    }

#define COSYVOICE_TTS_CONTEXT_GET_FLAG_IMPL(name, flag) \
    bool cosyvoice_tts_context_get_##name##_enabled(cosyvoice_tts_context_t ctx) \
    { \
        return (ctx->flags & flag) != 0; \
    }

#define COSYVOICE_TTS_CONTEXT_FLAG_ACCESSORS(name, flag) \
    COSYVOICE_TTS_CONTEXT_SET_FLAG_IMPL(name, flag) \
    COSYVOICE_TTS_CONTEXT_GET_FLAG_IMPL(name, flag)

bool cosyvoice_tts_context_set_text_normalization_enabled(cosyvoice_tts_context_t ctx, bool enabled)
{
#ifdef COSYVOICE_NO_ICU
    return !enabled;
#else
    set_flag(ctx, COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION, enabled);
    return true;
#endif
}

COSYVOICE_TTS_CONTEXT_GET_FLAG_IMPL(text_normalization, COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION)

COSYVOICE_TTS_CONTEXT_FLAG_ACCESSORS(split_text, COSYVOICE_TTS_FLAG_SPLIT_TEXT)

COSYVOICE_TTS_CONTEXT_FLAG_ACCESSORS(fast_split_text, COSYVOICE_TTS_FLAG_FAST_SPLIT)

COSYVOICE_TTS_CONTEXT_FLAG_ACCESSORS(fade_in, COSYVOICE_TTS_FLAG_FADE_IN)

uint32_t cosyvoice_tts_context_get_flags(cosyvoice_tts_context_t ctx)
{
    return ctx->flags;
}

uint32_t cosyvoice_tts_context_set_flags(cosyvoice_tts_context_t ctx, uint32_t flags)
{
    flags &= COSYVOICE_TTS_FLAG_MASK
#ifdef COSYVOICE_NO_ICU
        & ~COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION
#endif
        ;
    ctx->flags = flags;
    return ctx->flags;
}

bool cosyvoice_tts_zero_shot(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_generated_speech_ptr result)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_ZERO_SHOT, result, nullptr, nullptr);
}

bool cosyvoice_tts_instruct(cosyvoice_tts_context_t ctx, const char* text, const char* instruction, float speed, cosyvoice_generated_speech_ptr result)
{
    return ctx->tts_job(text, instruction, speed, COSYVOICE_INFERENCE_MODE_INSTRUCT, result, nullptr, nullptr);
}

bool cosyvoice_tts_cross_lingual(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_generated_speech_ptr result)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, result, nullptr, nullptr);
}

bool cosyvoice_tts_zero_shot_stream(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_tts_audio_callback_t callback, void* user_data)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_ZERO_SHOT, nullptr, callback, user_data);
}

bool cosyvoice_tts_instruct_stream(cosyvoice_tts_context_t ctx, const char* text, const char* instruction, float speed, cosyvoice_tts_audio_callback_t callback, void* user_data)
{
    return ctx->tts_job(text, instruction, speed, COSYVOICE_INFERENCE_MODE_INSTRUCT, nullptr, callback, user_data);
}

bool cosyvoice_tts_cross_lingual_stream(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_tts_audio_callback_t callback, void* user_data)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, nullptr, callback, user_data);
}
