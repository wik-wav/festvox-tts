/*
 * FestVox Festival batch wrapper with a measured, multi-epoch UniSyn join.
 *
 * This is a project-local extension of Festival 2.5.0. It deliberately keeps
 * stock source windows and target pitchmarks intact. Around an eligible unit
 * handoff it overlap-adds an outgoing and incoming source-frame trajectory
 * with complementary raised-cosine weights. The crossover is specified in
 * milliseconds; target epochs only snap its edges to complete PSOLA frames.
 *
 * Festival and Edinburgh Speech Tools license terms permit modification and
 * redistribution provided the original notices remain. This file is new
 * FestVox code and does not replace or modify the installed Festival binary.
 */

#include <festival.h>
#include <sigpr/EST_filter.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

typedef EST_TVector<EST_Wave> EST_WaveVector;
VAL_REGISTER_TYPE_DCLS(wavevector, EST_WaveVector)
VAL_REGISTER_TYPE_DCLS(ivector, EST_IVector)

void us_generate_wave(
    EST_Utterance &utt,
    const EST_String &filter_method,
    const EST_String &ola_method);
void add_wave_to_utterance(
    EST_Utterance &utt,
    EST_Wave &sig,
    const EST_String &name);

namespace {

const double kPi = 3.14159265358979323846;
const double kDefaultCrossoverMs = 40.0;
const double kMaximumCrossoverMs = 100.0;
const double kMinimumMixtureRetention = 0.70;

struct EpochBlend {
    bool active;
    int outgoing_frame;
    int incoming_frame;
    double outgoing_weight;
    double incoming_weight;
    bool lpc_coefficients_valid;
    EST_FVector lpc_coefficients;

    EpochBlend()
        : active(false),
          outgoing_frame(0),
          incoming_frame(0),
          outgoing_weight(1.0),
          incoming_weight(0.0),
          lpc_coefficients_valid(false)
    {
    }
};

struct JoinPlan {
    int unit_index;
    int source_boundary;
    int target_handoff;
    int target_start;
    int target_end;
    double requested_left_ms;
    double requested_right_ms;
    double context_cap_ms;
    double effective_ms;
    double minimum_retention;
    std::string phone;
    std::string context;
    std::string reason;
    bool active;

    JoinPlan()
        : unit_index(-1),
          source_boundary(0),
          target_handoff(0),
          target_start(0),
          target_end(0),
          requested_left_ms(0.0),
          requested_right_ms(0.0),
          context_cap_ms(0.0),
          effective_ms(0.0),
          minimum_retention(1.0),
          active(false)
    {
    }
};

struct FrameTrajectoryPlan {
    bool active;
    bool area_resampled;
    int centre_offset;
    double original_correlation;
    double corrected_correlation;
    std::vector<int> source_frames;
    std::vector<int> source_centre_offsets;
    std::vector<double> source_weights;
    std::string reason;

    FrameTrajectoryPlan()
        : active(false),
          area_resampled(false),
          centre_offset(0),
          original_correlation(0.0),
          corrected_correlation(0.0)
    {
    }
};

struct ContextPolicy {
    double maximum_ms;
    double phone_fraction;
    int minimum_intervals;
    std::string name;
};

double clamp_double(double value, double minimum, double maximum)
{
    if (!std::isfinite(value))
        return minimum;
    return std::max(minimum, std::min(maximum, value));
}

int clamp_int(int value, int minimum, int maximum)
{
    return std::max(minimum, std::min(maximum, value));
}

ContextPolicy context_policy(EST_Item *segment, const EST_String &phone)
{
    const std::string canonical = segment
        ? segment->S("festvox_join_context_class", "").str()
        : "";
    if (canonical == "silence")
        return ContextPolicy{0.0, 0.0, 0, "silence"};
    if (canonical == "vowel")
        return ContextPolicy{80.0, 0.60, 3, "vowel"};
    if (canonical == "nasal" || canonical == "liquid" ||
        canonical == "glide")
        return ContextPolicy{60.0, 0.70, 2, "sonorant"};
    if (canonical == "fricative_voiced")
        return ContextPolicy{50.0, 0.65, 2, "voiced-fricative"};
    if (canonical == "fricative_voiceless")
        return ContextPolicy{16.0, 0.25, 1, "unvoiced-fricative"};
    if (canonical == "stop_voiceless" || canonical == "stop_voiced" ||
        canonical == "affricate_voiceless" ||
        canonical == "affricate_voiced")
        return ContextPolicy{8.0, 0.20, 1, "closure-or-obstruent"};

    // Third-party voices without the explicit GUI annotation retain
    // Festival's phoneset-based behavior.
    if (ph_is_silence(phone))
        return ContextPolicy{0.0, 0.0, 0, "silence"};
    if (ph_is_vowel(phone))
        return ContextPolicy{80.0, 0.60, 3, "vowel"};
    if (ph_is_nasal(phone) || ph_is_liquid(phone) ||
        ph_is_approximant(phone))
        return ContextPolicy{60.0, 0.70, 2, "sonorant"};
    if (ph_is_fricative(phone))
        return ContextPolicy{
            ph_is_voiced(phone) ? 50.0 : 16.0,
            ph_is_voiced(phone) ? 0.65 : 0.25,
            ph_is_voiced(phone) ? 2 : 1,
            ph_is_voiced(phone) ? "voiced-fricative" :
                                  "unvoiced-fricative"};
    if (ph_is_stop(phone) || ph_is_obstruent(phone))
        return ContextPolicy{8.0, 0.20, 1, "closure-or-obstruent"};
    if (ph_is_sonorant(phone))
        return ContextPolicy{40.0, 0.40, 2, "other-sonorant"};
    return ContextPolicy{12.0, 0.25, 1, "other"};
}

bool supports_voiced_trajectory_correction(const ContextPolicy &policy)
{
    return (
        policy.name == "vowel" ||
        policy.name == "sonorant" ||
        policy.name == "voiced-fricative" ||
        policy.name == "other-sonorant");
}

int frame_centre(
    int frame_index,
    const EST_WaveVector &frames,
    const EST_IVector *frame_pm_indices)
{
    if (frame_pm_indices && frame_index >= 0 &&
        frame_index < frame_pm_indices->n())
        return frame_pm_indices->a_no_check(frame_index);
    return (frames(frame_index).num_samples() - 1) / 2;
}

double centred_frame_correlation(
    const EST_Wave &left,
    int left_centre,
    const EST_Wave &right,
    int right_centre,
    double *left_power_out,
    double *right_power_out)
{
    const int left_radius = std::min(
        left_centre, left.num_samples() - left_centre - 1);
    const int right_radius = std::min(
        right_centre, right.num_samples() - right_centre - 1);
    const int radius = std::min(96, std::min(left_radius, right_radius));
    if (radius < 4)
    {
        *left_power_out = 0.0;
        *right_power_out = 0.0;
        return 0.0;
    }

    const int count = 2 * radius + 1;
    double left_mean = 0.0;
    double right_mean = 0.0;
    for (int offset = -radius; offset <= radius; ++offset)
    {
        left_mean += left.a_no_check(left_centre + offset);
        right_mean += right.a_no_check(right_centre + offset);
    }
    left_mean /= count;
    right_mean /= count;

    double numerator = 0.0;
    double left_power = 0.0;
    double right_power = 0.0;
    for (int offset = -radius; offset <= radius; ++offset)
    {
        const double a =
            left.a_no_check(left_centre + offset) - left_mean;
        const double b =
            right.a_no_check(right_centre + offset) - right_mean;
        numerator += a * b;
        left_power += a * a;
        right_power += b * b;
    }
    *left_power_out = left_power;
    *right_power_out = right_power;
    const double denominator = std::sqrt(left_power * right_power);
    return denominator > 1.0e-12 ? numerator / denominator : 0.0;
}

struct PhaseAlignment {
    double zero_lag_correlation;
    double best_correlation;
    int best_lag;

    PhaseAlignment()
        : zero_lag_correlation(0.0),
          best_correlation(0.0),
          best_lag(0)
    {
    }
};

PhaseAlignment best_frame_alignment(
    const EST_Wave &left,
    int left_centre,
    const EST_Wave &right,
    int right_centre,
    int maximum_lag)
{
    PhaseAlignment result;
    double left_power = 0.0;
    double right_power = 0.0;
    result.zero_lag_correlation = centred_frame_correlation(
        left, left_centre, right, right_centre,
        &left_power, &right_power);
    result.best_correlation = result.zero_lag_correlation;
    maximum_lag = std::max(0, maximum_lag);
    for (int lag = -maximum_lag;
         lag <= maximum_lag;
         ++lag)
    {
        if (lag == 0)
            continue;
        const int shifted = right_centre + lag;
        if (shifted < 0 || shifted >= right.num_samples())
            continue;
        double trial_left_power = 0.0;
        double trial_right_power = 0.0;
        const double correlation = centred_frame_correlation(
            left, left_centre, right, shifted,
            &trial_left_power, &trial_right_power);
        if (correlation > result.best_correlation)
        {
            result.best_correlation = correlation;
            result.best_lag = lag;
        }
    }
    return result;
}

double predicted_mixture_retention(
    const EST_Wave &left,
    int left_centre,
    const EST_Wave &right,
    int right_centre,
    double left_weight,
    double right_weight)
{
    double left_power = 0.0;
    double right_power = 0.0;
    const double correlation = centred_frame_correlation(
        left, left_centre, right, right_centre,
        &left_power, &right_power);
    const double reference =
        left_weight * left_weight * left_power +
        right_weight * right_weight * right_power;
    if (reference <= 1.0e-12)
        return 1.0;
    const double cross =
        2.0 * left_weight * right_weight * correlation *
        std::sqrt(left_power * right_power);
    return std::sqrt(std::max(0.0, reference + cross) / reference);
}

double lpc_spectral_distance_db(
    const EST_Track &source,
    int left_frame,
    int right_frame)
{
    const int order = source.num_channels() - 1;
    if (order <= 0)
        return 0.0;

    // Compare the all-pole envelopes, not coefficient vectors. Predictor
    // coefficients are poorly scaled for Euclidean distance and channel zero
    // carries residual energy, which is reported independently below.
    const int bins = 24;
    double squared = 0.0;
    for (int bin = 1; bin <= bins; ++bin)
    {
        const double frequency =
            kPi * static_cast<double>(bin) /
            static_cast<double>(bins + 1);
        double left_real = 1.0;
        double left_imag = 0.0;
        double right_real = 1.0;
        double right_imag = 0.0;
        for (int coefficient = 1;
             coefficient <= order;
             ++coefficient)
        {
            const double phase = frequency * coefficient;
            const double cosine = std::cos(phase);
            const double sine = std::sin(phase);
            const double left =
                source.a_no_check(left_frame, coefficient);
            const double right =
                source.a_no_check(right_frame, coefficient);
            left_real -= left * cosine;
            left_imag += left * sine;
            right_real -= right * cosine;
            right_imag += right * sine;
        }
        const double left_magnitude = std::max(
            1.0e-12,
            std::sqrt(
                left_real * left_real +
                left_imag * left_imag));
        const double right_magnitude = std::max(
            1.0e-12,
            std::sqrt(
                right_real * right_real +
                right_imag * right_imag));
        const double difference =
            20.0 * std::log10(right_magnitude / left_magnitude);
        squared += difference * difference;
    }
    return std::sqrt(squared / bins);
}

EST_Item *segment_at_time(
    EST_Relation &segments,
    double when,
    double *start_out,
    double *end_out)
{
    double start = 0.0;
    for (EST_Item *segment = segments.head();
         segment;
         segment = inext(segment))
    {
        const double end = segment->F("end", start);
        if (when >= start - 1.0e-6 && when <= end + 1.0e-6)
        {
            *start_out = start;
            *end_out = end;
            return segment;
        }
        start = end;
    }
    return 0;
}

int first_target_at_or_after(const EST_Track &target, double when)
{
    for (int index = 0; index < target.num_frames(); ++index)
        if (target.t(index) >= when)
            return index;
    return std::max(0, target.num_frames() - 1);
}

int last_target_at_or_before(const EST_Track &target, double when)
{
    for (int index = target.num_frames() - 1; index >= 0; --index)
        if (target.t(index) <= when)
            return index;
    return 0;
}

int source_lerp(int first, int last, double progress)
{
    return clamp_int(
        static_cast<int>(std::floor(
            first + (last - first) * progress + 0.5)),
        std::min(first, last),
        std::max(first, last));
}

bool stable_lpc_blend(
    const EST_Track &source,
    int outgoing,
    int incoming,
    double outgoing_weight,
    double incoming_weight,
    EST_FVector *result)
{
    const int channels = source.num_channels();
    if (channels <= 0)
        return true;
    EST_FVector left_lpc(channels);
    EST_FVector right_lpc(channels);
    for (int channel = 0; channel < channels; ++channel)
    {
        left_lpc.a_no_check(channel) =
            source.a_no_check(outgoing, channel);
        right_lpc.a_no_check(channel) =
            source.a_no_check(incoming, channel);
    }

    const int order = channels - 1;
    std::vector<double> left_ref(channels, 0.0);
    std::vector<double> right_ref(channels, 0.0);
    std::vector<double> mixed_ref(channels, 0.0);
    std::vector<double> work(channels, 0.0);
    std::vector<double> next(channels, 0.0);

    // Festival stores predictor coefficients for
    // x[n] = residual[n] + sum(a_i*x[n-i]).  Schur recursion operates on
    // A(z)=1+sum(A_i*z^-i), hence A_i=-a_i here.  Debian's lpc2ref/ref2lpc
    // and LSF helpers abort as unfinished, so keep this small paired
    // conversion local and verify every input by round-tripping it.
    const auto lpc_to_reflection = [&](
            const EST_FVector &lpc,
            std::vector<double> *reflection) -> bool
    {
        std::fill(work.begin(), work.end(), 0.0);
        for (int index = 1; index <= order; ++index)
            work[index] = -static_cast<double>(
                lpc.a_no_check(index));
        for (int stage = order; stage >= 1; --stage)
        {
            const double k = work[stage];
            if (!std::isfinite(k) || std::fabs(k) >= 0.9995)
                return false;
            (*reflection)[stage] = k;
            const double denominator = 1.0 - k * k;
            if (denominator <= 1.0e-7)
                return false;
            std::fill(next.begin(), next.end(), 0.0);
            for (int index = 1; index < stage; ++index)
                next[index] = (
                    work[index] -
                    k * work[stage - index]) / denominator;
            work.swap(next);
        }
        (*reflection)[0] = std::max(
            1.0e-12, static_cast<double>(
                lpc.a_no_check(0)));
        return true;
    };
    const auto reflection_to_lpc = [&](
            const std::vector<double> &reflection,
            EST_FVector *lpc) -> bool
    {
        std::fill(work.begin(), work.end(), 0.0);
        for (int stage = 1; stage <= order; ++stage)
        {
            const double k = reflection[stage];
            if (!std::isfinite(k) || std::fabs(k) >= 0.9995)
                return false;
            std::fill(next.begin(), next.end(), 0.0);
            next[stage] = k;
            for (int index = 1; index < stage; ++index)
                next[index] = (
                    work[index] +
                    k * work[stage - index]);
            work.swap(next);
        }
        lpc->resize(channels);
        lpc->a_no_check(0) = static_cast<float>(
            reflection[0]);
        for (int index = 1; index <= order; ++index)
            lpc->a_no_check(index) = static_cast<float>(
                -work[index]);
        return true;
    };
    const auto round_trip_ok = [&](
            const EST_FVector &original,
            const std::vector<double> &reflection) -> bool
    {
        EST_FVector rebuilt;
        if (!reflection_to_lpc(reflection, &rebuilt))
            return false;
        for (int index = 1; index <= order; ++index)
        {
            const double expected = original.a_no_check(index);
            const double actual = rebuilt.a_no_check(index);
            const double scale = std::max(1.0, std::fabs(expected));
            if (!std::isfinite(actual) ||
                std::fabs(actual - expected) > 2.0e-4 * scale)
                return false;
        }
        return true;
    };
    if (!lpc_to_reflection(left_lpc, &left_ref) ||
        !lpc_to_reflection(right_lpc, &right_ref) ||
        !round_trip_ok(left_lpc, left_ref) ||
        !round_trip_ok(right_lpc, right_ref))
        return false;

    const double left_energy = std::max(
        1.0e-12, left_ref[0]);
    const double right_energy = std::max(
        1.0e-12, right_ref[0]);
    mixed_ref[0] = std::exp(
        outgoing_weight * std::log(left_energy) +
        incoming_weight * std::log(right_energy));
    for (int channel = 1; channel < channels; ++channel)
    {
        const double value =
            outgoing_weight * left_ref[channel] +
            incoming_weight * right_ref[channel];
        if (!std::isfinite(value))
            return false;
        mixed_ref[channel] = clamp_double(
            value, -0.999, 0.999);
    }
    return reflection_to_lpc(mixed_ref, result);
}

std::vector<JoinPlan> build_join_plans(
    EST_Utterance &utt,
    const EST_WaveVector &frames,
    const EST_Track &source,
    const EST_Track &target,
    const EST_IVector &map,
    const EST_IVector *frame_pm_indices,
    bool lpc,
    std::vector<EpochBlend> *epoch_blends)
{
    std::vector<JoinPlan> plans;
    EST_Relation *units = utt.relation("Unit", 1);
    EST_Relation *segments = utt.relation("Segment", 1);
    const double configured_ms = clamp_double(
        Param().F("festvox.join_crossover_ms",
                  static_cast<float>(kDefaultCrossoverMs)),
        0.0,
        kMaximumCrossoverMs);
    if (configured_ms <= 0.0 || !units || !segments)
        return plans;

    int source_boundary = 0;
    int unit_index = 0;
    for (EST_Item *unit = units->head();
         unit && inext(unit);
         unit = inext(unit), ++unit_index)
    {
        JoinPlan plan;
        plan.unit_index = unit_index;
        const int outgoing_count = unit->I("num_frames", 0);
        const int incoming_count = inext(unit)->I("num_frames", 0);
        source_boundary += outgoing_count;
        plan.source_boundary = source_boundary;
        if (outgoing_count <= 0 || incoming_count <= 0 ||
            source_boundary <= 0 ||
            source_boundary >= source.num_frames())
        {
            plan.reason = "insufficient-source-frames";
            plans.push_back(plan);
            continue;
        }

        int handoff = -1;
        for (int target_index = 1; target_index < map.n(); ++target_index)
        {
            if (map.a_no_check(target_index - 1) < source_boundary &&
                map.a_no_check(target_index) >= source_boundary)
            {
                handoff = target_index;
                break;
            }
        }
        if (handoff < 1 || handoff >= target.num_frames())
        {
            plan.reason = "unmapped-source-boundary";
            plans.push_back(plan);
            continue;
        }
        plan.target_handoff = handoff;
        const double handoff_time = target.t(handoff);
        double segment_start = 0.0;
        double segment_end = 0.0;
        EST_Item *shared_segment = segment_at_time(
            *segments, handoff_time, &segment_start, &segment_end);
        if (!shared_segment || segment_end <= segment_start)
        {
            plan.reason = "missing-shared-phone";
            plans.push_back(plan);
            continue;
        }
        const EST_String phone = shared_segment->S("name", "");
        plan.phone = phone.str();
        const ContextPolicy policy = context_policy(shared_segment, phone);
        plan.context = policy.name;
        if (policy.maximum_ms <= 0.0)
        {
            plan.reason = "silence-boundary";
            plans.push_back(plan);
            continue;
        }

        double requested_left = unit->F(
            "festvox_join_left_ms",
            static_cast<float>(configured_ms * 0.5));
        double requested_right = unit->F(
            "festvox_join_right_ms",
            static_cast<float>(configured_ms * 0.5));
        requested_left = clamp_double(
            requested_left, 0.0, kMaximumCrossoverMs);
        requested_right = clamp_double(
            requested_right, 0.0, kMaximumCrossoverMs);
        plan.requested_left_ms = requested_left;
        plan.requested_right_ms = requested_right;
        const double requested_total = requested_left + requested_right;
        if (requested_total <= 0.0)
        {
            plan.reason = "disabled";
            plans.push_back(plan);
            continue;
        }

        const double phone_ms = 1000.0 * (segment_end - segment_start);
        const double context_cap = std::min(
            policy.maximum_ms, policy.phone_fraction * phone_ms);
        plan.context_cap_ms = context_cap;
        const double effective_total = std::min(
            requested_total, context_cap);
        const double left_share = requested_left / requested_total;
        const double desired_left_ms = effective_total * left_share;
        const double desired_right_ms =
            effective_total - desired_left_ms;

        int start = first_target_at_or_after(
            target, std::max(
                segment_start,
                handoff_time - desired_left_ms / 1000.0));
        int end = last_target_at_or_before(
            target, std::min(
                segment_end,
                handoff_time + desired_right_ms / 1000.0));
        start = std::max(start, first_target_at_or_after(
            target, segment_start + 1.0e-6));
        end = std::min(end, last_target_at_or_before(
            target, segment_end - 1.0e-6));
        start = std::min(start, handoff - 1);
        end = std::max(end, handoff);

        // Do not cap this by an epoch/period count. At a higher F0 the same
        // perceptual duration legitimately contains more epochs. The global
        // millisecond limit and the shared-phone context limit already bound
        // both work and acoustic reach.
        if (start < 0 || end >= map.n() ||
            end >= target.num_frames() ||
            end - start < policy.minimum_intervals)
        {
            plan.reason = "insufficient-target-context";
            plans.push_back(plan);
            continue;
        }

        const int outgoing_first = clamp_int(
            map.a_no_check(start),
            source_boundary - outgoing_count,
            source_boundary - 1);
        const int outgoing_last = source_boundary - 1;
        const int incoming_first = source_boundary;
        const int incoming_last = clamp_int(
            map.a_no_check(end),
            source_boundary,
            source_boundary + incoming_count - 1);
        plan.target_start = start;
        plan.target_end = end;
        plan.effective_ms = 1000.0 *
            (target.t(end) - target.t(start));

        std::vector<EpochBlend> candidates(end - start + 1);
        bool safe = true;
        double minimum_retention = 1.0;
        for (int target_index = start;
             target_index <= end;
             ++target_index)
        {
            const double progress =
                static_cast<double>(target_index - start) /
                std::max(1, end - start);
            EpochBlend blend;
            blend.active = true;
            blend.outgoing_frame = source_lerp(
                outgoing_first, outgoing_last, progress);
            blend.incoming_frame = source_lerp(
                incoming_first, incoming_last, progress);
            blend.outgoing_weight =
                std::pow(std::cos(0.5 * kPi * progress), 2.0);
            blend.incoming_weight = 1.0 - blend.outgoing_weight;

            if (blend.outgoing_weight > 0.02 &&
                blend.incoming_weight > 0.02)
            {
                const int left_centre = frame_centre(
                    blend.outgoing_frame, frames, frame_pm_indices);
                const int right_centre = frame_centre(
                    blend.incoming_frame, frames, frame_pm_indices);
                const double retention = predicted_mixture_retention(
                    frames(blend.outgoing_frame),
                    left_centre,
                    frames(blend.incoming_frame),
                    right_centre,
                    blend.outgoing_weight,
                    blend.incoming_weight);
                minimum_retention = std::min(
                    minimum_retention, retention);
                if (retention < kMinimumMixtureRetention)
                    safe = false;
            }

            if (safe && lpc)
            {
                EST_FVector trial;
                if (!stable_lpc_blend(
                        source,
                        blend.outgoing_frame,
                        blend.incoming_frame,
                        blend.outgoing_weight,
                        blend.incoming_weight,
                        &trial))
                    safe = false;
                else
                {
                    // Validation computes the exact target coefficients.
                    // Retain them so the render pass does not repeat both
                    // Schur conversions and their round-trip checks.
                    blend.lpc_coefficients = trial;
                    blend.lpc_coefficients_valid = true;
                }
            }
            candidates[target_index - start] = blend;
        }
        plan.minimum_retention = minimum_retention;
        if (!safe)
        {
            plan.reason = lpc ? "unsafe-phase-or-lpc" :
                                "unsafe-phase-cancellation";
            plans.push_back(plan);
            continue;
        }

        bool collision = false;
        for (int target_index = start;
             target_index <= end;
             ++target_index)
            if ((*epoch_blends)[target_index].active)
                collision = true;
        if (collision)
        {
            plan.reason = "overlapping-crossover";
            plans.push_back(plan);
            continue;
        }
        for (int target_index = start;
             target_index <= end;
             ++target_index)
            (*epoch_blends)[target_index] =
                candidates[target_index - start];
        plan.active = true;
        plan.reason = (
            effective_total + 1.0e-6 < requested_total ?
            "context-capped" : "applied");
        plans.push_back(plan);
    }
    return plans;
}

void add_frame(
    EST_Wave &target,
    const EST_Wave &frame,
    int target_centre,
    int source_centre,
    double weight,
    bool exact)
{
    const int start = target_centre - source_centre;
    for (int sample = 0; sample < frame.num_samples(); ++sample)
    {
        const int target_sample = start + sample;
        if (target_sample < 0 ||
            target_sample >= target.num_samples())
            continue;
        if (exact)
        {
            target.a_no_check(target_sample) +=
                frame.a_no_check(sample);
        }
        else
        {
            const int contribution = static_cast<int>(std::floor(
                weight * frame.a_no_check(sample) +
                (weight >= 0.0 ? 0.5 : -0.5)));
            const int mixed = static_cast<int>(
                target.a_no_check(target_sample)) + contribution;
            target.a_no_check(target_sample) = static_cast<short>(
                clamp_int(mixed, -32768, 32767));
        }
    }
}

void map_or_blend_lpc(
    const EST_Track &source,
    EST_Track &target,
    const EST_IVector &map,
    const std::vector<EpochBlend> &blends)
{
    const int count = std::min(
        map.n(), target.num_frames());
    for (int frame = 0; frame < count; ++frame)
    {
        if (blends[frame].active)
        {
            if (blends[frame].lpc_coefficients_valid)
            {
                for (int channel = 0;
                     channel < target.num_channels();
                     ++channel)
                    target.a_no_check(frame, channel) =
                        blends[frame].lpc_coefficients.a_no_check(channel);
            }
            else
            {
                EST_FVector mixed;
                if (!stable_lpc_blend(
                        source,
                        blends[frame].outgoing_frame,
                        blends[frame].incoming_frame,
                        blends[frame].outgoing_weight,
                        blends[frame].incoming_weight,
                        &mixed))
                    EST_error("FestVox stable LPC crossover failed");
                for (int channel = 0;
                     channel < target.num_channels();
                     ++channel)
                    target.a_no_check(frame, channel) =
                        mixed.a_no_check(channel);
            }
        }
        else
        {
            const int source_frame = clamp_int(
                map.a_no_check(frame), 0,
                source.num_frames() - 1);
            for (int channel = 0;
                 channel < target.num_channels();
                 ++channel)
                target.a_no_check(frame, channel) =
                    source.a_no_check(source_frame, channel);
        }
    }
    for (int frame = count;
         frame < target.num_frames();
         ++frame)
        for (int channel = 0;
             channel < target.num_channels();
             ++channel)
            target.a_no_check(frame, channel) = 0.0;
}

std::vector<FrameTrajectoryPlan> build_frame_trajectory_plans(
    EST_Utterance &utt,
    const EST_WaveVector &frames,
    const EST_Track &source,
    const EST_Track &target,
    const EST_IVector &map,
    const EST_IVector *frame_pm_indices,
    const std::vector<EpochBlend> &blends,
    bool lpc)
{
    std::vector<FrameTrajectoryPlan> plans(map.n());
    if (Param().I("festvox.intra_unit_phase_correction", 1) == 0)
        return plans;

    EST_Relation *units = utt.relation("Unit", 1);
    EST_Relation *segments = utt.relation("Segment", 1);
    if (!units || !segments)
        return plans;

    std::vector<int> source_units(source.num_frames(), -1);
    std::vector<int> source_unit_starts;
    std::vector<int> source_unit_ends;
    int first = 0;
    int unit_index = 0;
    for (EST_Item *unit = units->head();
         unit;
         unit = inext(unit), ++unit_index)
    {
        const int frame_count = std::max(
            0, unit->I("num_frames", 0));
        source_unit_starts.push_back(first);
        const int end = std::min(
            source.num_frames(), first + frame_count);
        source_unit_ends.push_back(end);
        for (int frame = first; frame < end; ++frame)
            source_units[frame] = unit_index;
        first = end;
    }

    const int count = std::min(map.n(), target.num_frames());
    for (int target_index = 1;
         target_index < count;
         ++target_index)
    {
        if (blends[target_index].active ||
            blends[target_index - 1].active)
        {
            plans[target_index].reason = "unit-crossover";
            continue;
        }
        const int left_source = clamp_int(
            map.a_no_check(target_index - 1),
            0, source.num_frames() - 1);
        const int right_source = clamp_int(
            map.a_no_check(target_index),
            0, source.num_frames() - 1);
        const int source_unit = source_units[left_source];
        if (source_unit < 0 ||
            source_unit != source_units[right_source])
        {
            plans[target_index].reason = "source-unit-boundary";
            continue;
        }

        double segment_start = 0.0;
        double segment_end = 0.0;
        EST_Item *segment = segment_at_time(
            *segments,
            target.t(target_index),
            &segment_start,
            &segment_end);
        if (!segment)
        {
            plans[target_index].reason = "missing-segment";
            continue;
        }
        const EST_String phone = segment->S("name", "");
        const ContextPolicy policy = context_policy(segment, phone);
        if (!supports_voiced_trajectory_correction(policy))
        {
            plans[target_index].reason = "nonperiodic-context";
            continue;
        }

        const int left_centre =
            frame_centre(left_source, frames, frame_pm_indices) +
            plans[target_index - 1].centre_offset;
        const int right_centre =
            frame_centre(right_source, frames, frame_pm_indices);
        double left_power = 0.0;
        double right_power = 0.0;
        const double original_correlation = centred_frame_correlation(
            frames(left_source), left_centre,
            frames(right_source), right_centre,
            &left_power, &right_power);
        plans[target_index].original_correlation =
            original_correlation;
        if (original_correlation >= 0.82)
        {
            plans[target_index].corrected_correlation =
                original_correlation;
            plans[target_index].reason = "already-continuous";
            continue;
        }

        const int unit_start = (
            source_unit < static_cast<int>(source_unit_starts.size()) ?
            source_unit_starts[source_unit] : 0);
        double left_period_ms = 0.0;
        double right_period_ms = 0.0;
        if (left_source > unit_start)
            left_period_ms = 1000.0 * (
                source.t(left_source) -
                source.t(left_source - 1));
        if (right_source > unit_start)
            right_period_ms = 1000.0 * (
                source.t(right_source) -
                source.t(right_source - 1));
        const double reference_period_ms = std::max(
            1.0, std::min(
                left_period_ms > 0.0 ? left_period_ms : right_period_ms,
                right_period_ms > 0.0 ? right_period_ms : left_period_ms));
        const int maximum_lag = std::max(
            1, static_cast<int>(std::floor(
                frames(right_source).sample_rate() *
                reference_period_ms * 0.00025 + 0.5)));
        const PhaseAlignment alignment = best_frame_alignment(
            frames(left_source), left_centre,
            frames(right_source), right_centre,
            maximum_lag);
        plans[target_index].corrected_correlation =
            alignment.best_correlation;
        if (alignment.best_lag != 0 &&
            alignment.best_correlation >= 0.90 &&
            alignment.best_correlation -
                original_correlation >= 0.15)
        {
            plans[target_index].active = true;
            plans[target_index].centre_offset = alignment.best_lag;
            plans[target_index].reason = "phase-reference-corrected";
            continue;
        }

        // Area resampling is deliberately experimental. In real Lem material
        // it changed a wider acoustic neighborhood than the diagnosed epoch
        // and could increase broadband novelty beside a splice. Keep normal
        // rendering limited to the directly measured phase correction above.
        if (Param().I("festvox.intra_unit_area_resampling", 0) == 0)
        {
            plans[target_index].reason = "no-safe-phase-recovery";
            continue;
        }

        const double level_step_db = (
            left_power > 1.0e-12 && right_power > 1.0e-12 ?
            10.0 * std::log10(right_power / left_power) : 0.0);
        if (lpc ||
            right_source <= left_source ||
            alignment.best_correlation < 0.70 ||
            (original_correlation >= 0.82 &&
             std::fabs(level_step_db) < 2.0))
        {
            plans[target_index].reason = "no-safe-phase-recovery";
            continue;
        }
        bool crossover_nearby = false;
        for (int offset = -4; offset <= 4; ++offset)
        {
            const int neighbor = target_index + offset;
            if (neighbor >= 0 &&
                neighbor < static_cast<int>(blends.size()) &&
                blends[neighbor].active)
                crossover_nearby = true;
        }
        if (crossover_nearby)
        {
            plans[target_index].reason =
                "crossover-neighborhood";
            continue;
        }

        const int unit_end = (
            source_unit < static_cast<int>(source_unit_ends.size()) ?
            source_unit_ends[source_unit] : source.num_frames());
        const int previous_source = (
            target_index >= 1 ? left_source : right_source);
        int next_source = right_source;
        if (target_index + 1 < count)
        {
            const int candidate = clamp_int(
                map.a_no_check(target_index + 1),
                0, source.num_frames() - 1);
            if (source_units[candidate] == source_unit)
                next_source = candidate;
        }
        const double left_boundary = (
            source_units[previous_source] == source_unit ?
            0.5 * (previous_source + right_source) :
            right_source - 0.5);
        const double right_boundary = (
            source_units[next_source] == source_unit ?
            0.5 * (right_source + next_source) :
            right_source + 0.5);
        int area_first = std::max(
            unit_start,
            static_cast<int>(std::floor(left_boundary)));
        int area_last = std::min(
            unit_end - 1,
            static_cast<int>(std::ceil(right_boundary)));
        // Keep runtime bounded even under extreme source compression.
        area_first = std::max(area_first, right_source - 4);
        area_last = std::min(area_last, right_source + 4);
        if (area_last - area_first + 1 < 2)
        {
            plans[target_index].reason =
                "insufficient-area-support";
            continue;
        }

        std::vector<int> candidate_frames;
        std::vector<int> candidate_offsets;
        std::vector<double> candidate_weights;
        const int raw_count = area_last - area_first + 1;
        double weight_sum = 0.0;
        for (int source_frame = area_first;
             source_frame <= area_last;
             ++source_frame)
        {
            const int position = source_frame - area_first;
            const double raw_weight = std::pow(
                std::sin(
                    kPi * static_cast<double>(position + 1) /
                    static_cast<double>(raw_count + 1)),
                2.0);
            int centre_offset = 0;
            if (source_frame != right_source)
            {
                double candidate_period_ms = reference_period_ms;
                if (source_frame > unit_start)
                    candidate_period_ms = 1000.0 * (
                        source.t(source_frame) -
                        source.t(source_frame - 1));
                const int candidate_maximum_lag = std::max(
                    1, static_cast<int>(std::floor(
                        frames(source_frame).sample_rate() *
                        std::max(
                            1.0,
                            std::min(
                                reference_period_ms,
                                candidate_period_ms)) *
                        0.00025 + 0.5)));
                const PhaseAlignment relative =
                    best_frame_alignment(
                        frames(right_source), right_centre,
                        frames(source_frame),
                        frame_centre(
                            source_frame,
                            frames,
                            frame_pm_indices),
                        candidate_maximum_lag);
                if (relative.best_correlation < 0.70)
                    continue;
                if (relative.best_lag != 0 &&
                    relative.best_correlation -
                        relative.zero_lag_correlation >= 0.10)
                    centre_offset = relative.best_lag;
            }
            candidate_frames.push_back(source_frame);
            candidate_offsets.push_back(centre_offset);
            candidate_weights.push_back(raw_weight);
            weight_sum += raw_weight;
        }
        if (candidate_frames.size() < 2 ||
            weight_sum <= 1.0e-12)
        {
            plans[target_index].reason =
                "unsafe-area-support";
            continue;
        }
        for (std::vector<double>::iterator weight =
                 candidate_weights.begin();
             weight != candidate_weights.end();
             ++weight)
            *weight /= weight_sum;
        plans[target_index].active = true;
        plans[target_index].area_resampled = true;
        plans[target_index].source_frames = candidate_frames;
        plans[target_index].source_centre_offsets =
            candidate_offsets;
        plans[target_index].source_weights = candidate_weights;
        plans[target_index].reason =
            "compressed-source-area-resampled";
    }

    // Do not let a broad source-area estimate alter the neighborhood of a
    // stronger measured phase correction.
    for (int index = 1; index < count; ++index)
    {
        if (!plans[index].area_resampled)
            continue;
        bool phase_nearby = false;
        for (int offset = -2; offset <= 2; ++offset)
        {
            const int neighbor = index + offset;
            if (neighbor >= 0 && neighbor < count &&
                plans[neighbor].active &&
                !plans[neighbor].area_resampled)
                phase_nearby = true;
        }
        if (phase_nearby)
        {
            plans[index].active = false;
            plans[index].area_resampled = false;
            plans[index].source_frames.clear();
            plans[index].source_centre_offsets.clear();
            plans[index].source_weights.clear();
            plans[index].reason =
                "phase-correction-neighborhood";
        }
    }

    // Never hand a shifted source phase directly to the separate unit
    // crossover trajectory. That transition produced a broadband impulse in
    // real Lem material even though the within-unit correlation improved.
    // Reject the whole contiguous correction run when it touches a crossover;
    // the crossover remains the sole owner of that acoustic neighborhood.
    int run_start = 1;
    while (run_start < count)
    {
        while (run_start < count &&
               (!plans[run_start].active ||
                plans[run_start].area_resampled))
            ++run_start;
        if (run_start >= count)
            break;
        int run_end = run_start;
        while (run_end + 1 < count &&
               plans[run_end + 1].active &&
               !plans[run_end + 1].area_resampled)
            ++run_end;
        const bool touches_left = (
            run_start > 0 && blends[run_start - 1].active);
        const bool touches_right = (
            run_end + 1 < count && blends[run_end + 1].active);
        const int run_length = run_end - run_start + 1;
        if ((touches_left || touches_right) &&
            run_length <= 4)
        {
            for (int index = run_start;
                 index <= run_end;
                 ++index)
            {
                plans[index].active = false;
                plans[index].centre_offset = 0;
                plans[index].reason = "crossover-neighborhood";
            }
        }
        else
        {
            if (touches_left)
            {
                plans[run_start].centre_offset =
                    static_cast<int>(std::round(
                        plans[run_start].centre_offset / 3.0));
                plans[run_start].reason =
                    "crossover-edge-taper";
                if (run_length >= 2)
                {
                    plans[run_start + 1].centre_offset =
                        static_cast<int>(std::round(
                            2.0 *
                            plans[run_start + 1].centre_offset /
                            3.0));
                    plans[run_start + 1].reason =
                        "crossover-edge-taper";
                }
            }
            if (touches_right)
            {
                plans[run_end].centre_offset =
                    static_cast<int>(std::round(
                        plans[run_end].centre_offset / 3.0));
                plans[run_end].reason =
                    "crossover-edge-taper";
                if (run_length >= 2)
                {
                    plans[run_end - 1].centre_offset =
                        static_cast<int>(std::round(
                            2.0 *
                            plans[run_end - 1].centre_offset /
                            3.0));
                    plans[run_end - 1].reason =
                        "crossover-edge-taper";
                }
            }
        }
        run_start = run_end + 1;
    }

    return plans;
}

void print_epoch_diagnostics(
    EST_Utterance &utt,
    const EST_WaveVector &frames,
    const EST_Track &source,
    const EST_Track &target,
    const EST_IVector &map,
    const EST_IVector *frame_pm_indices,
    const std::vector<EpochBlend> &blends,
    const std::vector<FrameTrajectoryPlan> &trajectory_plans,
    bool lpc)
{
    if (Param().I("festvox.debug_epoch_diagnostics", 0) == 0)
        return;

    EST_Relation *units = utt.relation("Unit", 1);
    EST_Relation *segments = utt.relation("Segment", 1);
    if (!units || !segments)
        return;

    std::vector<int> source_units(source.num_frames(), -1);
    std::vector<int> source_unit_starts;
    int first = 0;
    int unit_index = 0;
    for (EST_Item *unit = units->head();
         unit;
         unit = inext(unit), ++unit_index)
    {
        const int count = std::max(0, unit->I("num_frames", 0));
        source_unit_starts.push_back(first);
        const int end = std::min(
            source.num_frames(), first + count);
        for (int frame = first; frame < end; ++frame)
            source_units[frame] = unit_index;
        first = end;
    }

    const int count = std::min(map.n(), target.num_frames());
    for (int target_index = 1;
         target_index < count;
         ++target_index)
    {
        const int left_source = clamp_int(
            map.a_no_check(target_index - 1),
            0, source.num_frames() - 1);
        const int right_source = clamp_int(
            map.a_no_check(target_index),
            0, source.num_frames() - 1);
        const int source_unit = source_units[left_source];
        if (source_unit < 0 ||
            source_unit != source_units[right_source])
            continue;

        int previous_step = right_source - left_source;
        if (target_index >= 2)
        {
            const int previous_source = clamp_int(
                map.a_no_check(target_index - 2),
                0, source.num_frames() - 1);
            if (source_units[previous_source] == source_unit)
                previous_step = left_source - previous_source;
        }

        double left_power = 0.0;
        double right_power = 0.0;
        const double correlation = centred_frame_correlation(
            frames(left_source),
            frame_centre(left_source, frames, frame_pm_indices),
            frames(right_source),
            frame_centre(right_source, frames, frame_pm_indices),
            &left_power,
            &right_power);
        double level_step_db = 0.0;
        if (left_power > 1.0e-12 && right_power > 1.0e-12)
            level_step_db =
                10.0 * std::log10(right_power / left_power);

        const int unit_start = (
            source_unit < static_cast<int>(source_unit_starts.size()) ?
            source_unit_starts[source_unit] : 0);
        double left_period_ms = 0.0;
        double right_period_ms = 0.0;
        if (left_source > unit_start)
            left_period_ms = 1000.0 * (
                source.t(left_source) -
                source.t(left_source - 1));
        if (right_source > unit_start)
            right_period_ms = 1000.0 * (
                source.t(right_source) -
                source.t(right_source - 1));
        const double reference_period_ms = std::max(
            1.0, std::min(
                left_period_ms > 0.0 ? left_period_ms : right_period_ms,
                right_period_ms > 0.0 ? right_period_ms : left_period_ms));
        const int maximum_lag = std::max(
            1, static_cast<int>(std::floor(
                frames(right_source).sample_rate() *
                reference_period_ms * 0.00025 + 0.5)));
        const PhaseAlignment alignment = best_frame_alignment(
            frames(left_source),
            frame_centre(left_source, frames, frame_pm_indices),
            frames(right_source),
            frame_centre(right_source, frames, frame_pm_indices),
            maximum_lag);

        double segment_start = 0.0;
        double segment_end = 0.0;
        EST_Item *segment = segment_at_time(
            *segments,
            target.t(target_index),
            &segment_start,
            &segment_end);
        const std::string phone = segment ?
            segment->S("name", "").str() : "";
        const double spectral_distance = lpc ?
            lpc_spectral_distance_db(
                source, left_source, right_source) : 0.0;
        std::cout
            << "(GUIEPOCH "
            << target_index << " "
            << target.t(target_index) << " "
            << left_source << " "
            << right_source << " "
            << (right_source - left_source) << " "
            << ((right_source - left_source) - previous_step) << " "
            << correlation << " "
            << alignment.best_correlation << " "
            << alignment.best_lag << " "
            << level_step_db << " "
            << spectral_distance << " "
            << left_period_ms << " "
            << right_period_ms << " "
            << (blends[target_index].active ? 1 : 0) << " "
            << (trajectory_plans[target_index].active ? 1 : 0) << " "
            << trajectory_plans[target_index].centre_offset << " "
            << (trajectory_plans[target_index].area_resampled ? 1 : 0)
            << " "
            << trajectory_plans[target_index].source_frames.size() << " "
            << "\"" << phone << "\")"
            << std::endl;
    }
}

void print_frame_trajectory_plans(
    const std::vector<FrameTrajectoryPlan> &plans,
    const EST_Track &target,
    const EST_IVector &map)
{
    for (int index = 1;
         index < static_cast<int>(plans.size()) &&
         index < target.num_frames() &&
         index < map.n();
         ++index)
    {
        const FrameTrajectoryPlan &plan = plans[index];
        if (!plan.active)
            continue;
        std::cout
            << "(GUIFRAMEFIX "
            << index << " "
            << target.t(index) << " "
            << map.a_no_check(index - 1) << " "
            << map.a_no_check(index) << " "
            << plan.centre_offset << " "
            << plan.original_correlation << " "
            << plan.corrected_correlation << " "
            << (plan.area_resampled ? 1 : 0) << " "
            << plan.source_frames.size() << " "
            << "\"" << plan.reason << "\")"
            << std::endl;
    }
}

void print_join_plans(
    const std::vector<JoinPlan> &plans,
    const EST_Track &target)
{
    for (std::vector<JoinPlan>::const_iterator plan = plans.begin();
         plan != plans.end();
         ++plan)
    {
        const double start_time =
            plan->target_start >= 0 &&
            plan->target_start < target.num_frames() ?
            target.t(plan->target_start) : 0.0;
        const double end_time =
            plan->target_end >= 0 &&
            plan->target_end < target.num_frames() ?
            target.t(plan->target_end) : 0.0;
        std::cout
            << "(GUIXOVER "
            << plan->unit_index << " "
            << plan->target_handoff << " "
            << plan->target_start << " "
            << plan->target_end << " "
            << start_time << " "
            << end_time << " "
            << plan->requested_left_ms << " "
            << plan->requested_right_ms << " "
            << plan->context_cap_ms << " "
            << plan->effective_ms << " "
            << plan->minimum_retention << " "
            << (plan->active ? 1 : 0) << " "
            << "\"" << plan->phone << "\" "
            << "\"" << plan->context << "\" "
            << "\"" << plan->reason << "\")"
            << std::endl;
    }
}

LISP festvox_us_generate_wave(LISP lutt, LISP lfilter)
{
    EST_Utterance *utt = get_c_utt(lutt);
    const EST_String filter_method = get_c_string(lfilter);
    const bool lpc = filter_method == "lpc";
    const bool symmetric =
        Param().I("unisyn.window_symmetric", 1) != 0;
    EST_Item *source_item =
        utt->relation("SourceCoef", 1)->head();
    EST_WaveVector *frames =
        wavevector(source_item->f("frame"));
    EST_Track *source =
        track(source_item->f("coefs"));
    EST_Track *target =
        track(utt->relation("TargetCoef", 1)->head()->f("coefs"));
    EST_IVector *map =
        ivector(utt->relation("US_map", 1)->head()->f("map"));
    EST_IVector *frame_pm_indices = 0;
    if (!symmetric && source_item->f_present("pm_indices"))
        frame_pm_indices = ivector(
            source_item->f("pm_indices"));

    if (frames->length() <= 0 || map->n() <= 0)
    {
        us_generate_wave(
            *utt, filter_method,
            symmetric ? "analysis_period" :
                        "asymmetric_window");
        return lutt;
    }

    std::vector<EpochBlend> blends(map->n());
    const std::vector<JoinPlan> plans = build_join_plans(
        *utt, *frames, *source, *target, *map,
        frame_pm_indices, lpc, &blends);
    const std::vector<FrameTrajectoryPlan> trajectory_plans =
        build_frame_trajectory_plans(
            *utt, *frames, *source, *target, *map,
            frame_pm_indices, blends, lpc);
    bool has_active = false;
    for (std::vector<JoinPlan>::const_iterator plan = plans.begin();
         plan != plans.end();
         ++plan)
        has_active = has_active || plan->active;
    bool has_trajectory_correction = false;
    for (std::vector<FrameTrajectoryPlan>::const_iterator plan =
             trajectory_plans.begin();
         plan != trajectory_plans.end();
         ++plan)
        has_trajectory_correction =
            has_trajectory_correction || plan->active;
    print_join_plans(plans, *target);
    print_frame_trajectory_plans(
        trajectory_plans, *target, *map);
    print_epoch_diagnostics(
        *utt, *frames, *source, *target, *map,
        frame_pm_indices, blends, trajectory_plans, lpc);
    if (!has_active && !has_trajectory_correction)
    {
        us_generate_wave(
            *utt, filter_method,
            symmetric ? "analysis_period" :
                        "asymmetric_window");
        return lutt;
    }

    const int sample_rate = (*frames)(0).sample_rate();
    const int final_map = clamp_int(
        map->a_no_check(map->n() - 1),
        0, frames->length() - 1);
    const int final_offset = (
        !trajectory_plans.empty() ?
        trajectory_plans[
            std::min(
                static_cast<int>(trajectory_plans.size()) - 1,
                map->n() - 1)].centre_offset : 0);
    int last_sample = 0;
    if (symmetric)
    {
        last_sample = static_cast<int>(
            std::floor(target->end() * sample_rate + 0.5)) +
            (((*frames)(final_map).num_samples() - 1) / 2);
    }
    else
    {
        const int final_centre = frame_centre(
            final_map, *frames, frame_pm_indices) + final_offset;
        last_sample = static_cast<int>(
            std::floor(target->end() * sample_rate + 0.5)) +
            ((*frames)(final_map).num_samples() -
             final_centre - 1);
    }

    EST_Wave *signal = new EST_Wave;
    signal->resize(last_sample + 1);
    signal->fill(0);
    signal->set_sample_rate(sample_rate);
    const int frame_count = std::min(
        map->n(), target->num_frames());
    for (int target_index = 0;
         target_index < frame_count;
         ++target_index)
    {
        const int target_centre = static_cast<int>(
            std::floor(
                target->t(target_index) * sample_rate + 0.5));
        if (blends[target_index].active)
        {
            const EpochBlend &blend = blends[target_index];
            add_frame(
                *signal,
                (*frames)(blend.outgoing_frame),
                target_centre,
                frame_centre(
                    blend.outgoing_frame,
                    *frames,
                    frame_pm_indices),
                blend.outgoing_weight,
                false);
            add_frame(
                *signal,
                (*frames)(blend.incoming_frame),
                target_centre,
                frame_centre(
                    blend.incoming_frame,
                    *frames,
                    frame_pm_indices),
                blend.incoming_weight,
                false);
        }
        else
        {
            const int source_frame = clamp_int(
                map->a_no_check(target_index),
                0, frames->length() - 1);
            const FrameTrajectoryPlan &trajectory =
                trajectory_plans[target_index];
            if (trajectory.area_resampled &&
                trajectory.source_frames.size() ==
                    trajectory.source_centre_offsets.size() &&
                trajectory.source_frames.size() ==
                    trajectory.source_weights.size())
            {
                for (int contribution = 0;
                     contribution < static_cast<int>(
                         trajectory.source_frames.size());
                     ++contribution)
                {
                    const int contribution_frame = clamp_int(
                        trajectory.source_frames[contribution],
                        0, frames->length() - 1);
                    add_frame(
                        *signal,
                        (*frames)(contribution_frame),
                        target_centre,
                        frame_centre(
                            contribution_frame,
                            *frames,
                            frame_pm_indices) +
                            trajectory.source_centre_offsets[
                                contribution],
                        trajectory.source_weights[contribution],
                        false);
                }
            }
            else
            {
                add_frame(
                    *signal,
                    (*frames)(source_frame),
                    target_centre,
                    frame_centre(
                        source_frame,
                        *frames,
                        frame_pm_indices) +
                        trajectory.centre_offset,
                    1.0,
                    true);
            }
        }
    }

    if (lpc)
    {
        map_or_blend_lpc(
            *source, *target, *map, blends);
        EST_Wave *residual = new EST_Wave;
        residual->copy(*signal);
        utt->relation("TargetCoef", 1)->head()->set_val(
            "residual", est_val(residual));
        lpc_filter_fast(*target, *signal, *signal);
    }
    add_wave_to_utterance(*utt, *signal, "Wave");
    return lutt;
}

}  // namespace

void initialize_festvox_festival()
{
    festival_initialize(1, FESTIVAL_HEAP_SIZE);
    init_subr_2(
        "festvox_us_generate_wave",
        festvox_us_generate_wave,
        "(festvox_us_generate_wave UTT FILTER_METHOD)\n"
        "Use the project-local bounded multi-epoch UniSyn crossover.");
}

int main(int argc, char **argv)
{
    if (argc == 2 && EST_String(argv[1]) == "--server")
    {
        initialize_festvox_festival();
        std::cout << "FESTVOX-NATIVE-SERVER-READY" << std::endl;
        std::string request;
        while (std::getline(std::cin, request))
        {
            if (!request.empty() && request[request.size() - 1] == '\r')
                request.erase(request.size() - 1);
            if (request == "QUIT")
                break;
            const std::string::size_type separator = request.find('\t');
            if (separator == std::string::npos ||
                separator == 0 ||
                separator + 1 >= request.size())
            {
                std::cout
                    << "(GUIJOBEND \"invalid-request\" 2)"
                    << std::endl;
                continue;
            }
            const std::string token = request.substr(0, separator);
            const std::string script = request.substr(separator + 1);
            std::cout
                << "(GUIJOBBEGIN \"" << token << "\")"
                << std::endl;
            const int loaded = festival_load_file(script.c_str());
            std::cout
                << "(GUIJOBEND \"" << token << "\" "
                << (loaded ? 0 : 1) << ")"
                << std::endl;
        }
        festival_tidy_up();
        return 0;
    }
    if (argc != 3 || EST_String(argv[1]) != "-b")
    {
        std::cerr
            << "usage: festvox-festival -b SCRIPT.scm\n"
            << "       festvox-festival --server\n";
        return 2;
    }
    initialize_festvox_festival();
    const int loaded = festival_load_file(argv[2]);
    festival_tidy_up();
    return loaded ? 0 : 1;
}
