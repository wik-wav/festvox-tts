import copy
import importlib.util
import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import festvox_core as fc
fg = None
if importlib.util.find_spec("PyQt5") and importlib.util.find_spec("pyqtgraph"):
    import festvox_gui as fg


@unittest.skipIf(fg is None, "PyQt5/pyqtgraph are not installed in this runtime")
class MainWindowSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = fg.QtWidgets.QApplication.instance() or \
            fg.QtWidgets.QApplication([])

    def setUp(self):
        self._real_config_path = fg.CONFIG_PATH
        self._config_tmp = tempfile.TemporaryDirectory()
        fg.CONFIG_PATH = os.path.join(self._config_tmp.name, "config.json")
        cfg = copy.deepcopy(fc.DEFAULT_CONFIG)
        cfg["engine"] = "diphone"
        self.window = fg.MainWindow(cfg)

    def _rendered_sentence_with_unit_choices(self, text):
        segments = [
            fc.Segment("a", 0.0, 0.1),
            fc.Segment("b", 0.1, 0.2),
            fc.Segment("c", 0.2, 0.3),
        ]
        synthesis = fc.Synthesis(
            np.linspace(-0.1, 0.1, 300, dtype=np.float32),
            1000,
            segments,
            text=text,
            phones=["a", "b", "c"],
            diphones=["a-b", "b-c"],
            selected_units={0: "a__context", 1: "b__manual"},
            unit_overrides={1: "b__manual"},
        )
        state = self.window._new_sentence_state(text)
        state.update({
            "rendered_text": text,
            "synthesis": synthesis,
            "editor_segments": copy.deepcopy(segments),
            "preview_audio": synthesis.samples.copy(),
            "preview_sr": synthesis.sr,
            "rendered": True,
        })
        return state

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        fg.QtCore.QCoreApplication.sendPostedEvents(
            None, fg.QtCore.QEvent.DeferredDelete)
        self.app.processEvents()
        fg.CONFIG_PATH = self._real_config_path
        self._config_tmp.cleanup()

    def test_project_launch_argument_is_removed_before_qt_starts(self):
        launch, qt_arguments = fg.parse_launch_args([
            "--project", r"C:\reviews\project.json", "-style", "Fusion",
        ])

        self.assertEqual(launch.project, r"C:\reviews\project.json")
        self.assertEqual(qt_arguments, ["-style", "Fusion"])

    def test_parameter_editor_and_fault_menu_exist(self):
        self.assertEqual(self.window.undo_stack.undoLimit(), 64)
        self.assertTrue(self.window.action_show_rendered_joins.isCheckable())
        self.assertFalse(self.window.action_show_rendered_joins.isChecked())
        self.assertEqual(self.window.parameter_stack.count(), 8)
        self.assertEqual(
            [self.window.parameter_mode.itemData(index)
             for index in range(self.window.parameter_mode.count())],
            ["timing", "pitch", "voicing", "vocal_tract", "intonation",
             "recordings", "japanese", "mora_voicing"])
        self.assertEqual(
            self.window.parameter_mode.itemText(
                self.window.parameter_mode.findData("japanese")),
            "Pitch accent")
        self.assertEqual(
            self.window.curve_unit_overlay.text(),
            "Show syllables / morae")
        self.assertFalse(self.window.curve_unit_overlay.isChecked())
        self.assertEqual(set(self.window.fault_actions), {
            "disable_phone_timing", "disable_prosody",
            "disable_f0_correction", "single_pause", "pitch_glitch",
            "no_sustain_stretch", "legacy_joins", "monotone",
        })
        self.assertEqual(
            self.window.fault_actions["legacy_joins"].text(),
            "Legacy joins",
        )
        self.window._update_fault_availability()
        self.assertTrue(
            self.window.fault_actions["legacy_joins"].isVisible())
        self.assertEqual(set(self.window.bit_depth_actions), {0, 1, 2, 4, 8})
        self.assertEqual(self.window.editor_splitter.orientation(),
                         fg.Qt.Vertical)
        self.assertEqual(len(self.window.sentences), 1)

    def test_playback_uses_one_cancellable_finish_timer(self):
        class NullPlayer:
            mode = "test"

            def play(self, _samples, _rate):
                pass

            def stop(self):
                pass

            def shutdown(self):
                pass

        self.window.player.shutdown()
        self.window.player = NullPlayer()
        self.window._start_playback(np.zeros(16000, np.float32), 16000)
        self.assertTrue(self.window._playback_finish_timer.isActive())
        self.window._start_playback(np.zeros(32000, np.float32), 16000)
        self.assertTrue(self.window._playback_finish_timer.isActive())
        self.window.on_stop()
        self.assertFalse(self.window._playback_finish_timer.isActive())

    def test_player_removes_owned_temporary_files(self):
        player = fg.Player()
        handle = tempfile.NamedTemporaryFile(delete=False)
        path = handle.name
        handle.close()
        player._tmp = path
        player._temp_paths.add(path)

        player._cleanup_temp()

        self.assertFalse(Path(path).exists())
        self.assertFalse(player._temp_paths)
        player.shutdown()

    def test_synthesis_worker_keeps_qt_event_loop_responsive(self):
        events = []
        fg.QtCore.QTimer.singleShot(
            10, lambda: events.append((
                self.window._synthesis_busy,
                self.window.btn_gen.isEnabled(),
                self.window.sentences_view.isEnabled(),
                self.window.mode_tabs.tabBar().isEnabled(),
                fg.QtWidgets.QApplication.overrideCursor() is None,
            )))

        def slow_result():
            time.sleep(0.08)
            return "rendered"

        result = self.window._run_synthesis_task(slow_result)

        self.assertEqual(result, "rendered")
        self.assertEqual(events, [(True, False, True, True, True)])
        self.assertFalse(self.window._synthesis_busy)

    def test_synthesis_worker_propagates_backend_error_and_unlocks_ui(self):
        def fail():
            raise fc.BackendError("fixture failure")

        with self.assertRaisesRegex(fc.BackendError, "fixture failure"):
            self.window._run_synthesis_task(fail)

        self.assertFalse(self.window._synthesis_busy)
        self.assertTrue(self.window.btn_gen.isEnabled())

    def test_shutdown_resources_is_idempotent(self):
        with mock.patch.object(
                self.window.player, "shutdown") as player_shutdown, \
                mock.patch.object(
                    self.window.fest, "shutdown") as fest_shutdown, \
                mock.patch.object(self.window, "_persist_config") as persist:
            self.window._shutdown_resources()
            self.window._shutdown_resources()

        player_shutdown.assert_called_once_with()
        fest_shutdown.assert_called_once_with()
        persist.assert_called_once_with()

    def test_busy_close_is_deferred_instead_of_abandoning_process(self):
        event = fg.QtGui.QCloseEvent()
        self.window._synthesis_busy = True

        self.window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        self.assertTrue(self.window._close_requested)
        self.assertTrue(self.window._batch_cancel_requested)
        self.window._synthesis_busy = False
        self.window._close_requested = False

    def test_single_generation_shows_only_current_progress(self):
        observed = []

        def generate(**_kwargs):
            observed.append({
                "current": not self.window.generation_progress.isHidden(),
                "total": not self.window.batch_progress.isHidden(),
                "stack": not self.window.synthesis_progress_stack.isHidden(),
                "range": (
                    self.window.generation_progress.minimum(),
                    self.window.generation_progress.maximum(),
                ),
                "label": self.window.generation_progress.format(),
                "height": self.window.generation_progress.height(),
                "stack_height":
                    self.window.synthesis_progress_stack.height(),
            })
            return "rendered"

        with mock.patch.object(
                self.window, "_generate_current", side_effect=generate):
            result = self.window._generate_for_sentence_mode()

        self.assertEqual(result, "rendered")
        self.assertEqual(observed, [{
            "current": True,
            "total": False,
            "stack": True,
            "range": (0, 0),
            "label": "Generating current sentence...",
            "height": self.window._synthesis_progress_height,
            "stack_height": self.window._synthesis_progress_height,
        }])
        self.assertTrue(self.window.generation_progress.isHidden())
        self.assertTrue(self.window.synthesis_progress_stack.isHidden())

    def test_batch_generation_stacks_current_over_total_progress(self):
        self.window._begin_sentence_batch(3, "Generate")
        self.window._update_sentence_batch(1, 3, "Generate")
        observed = []

        def generate(**_kwargs):
            layout = self.window.synthesis_progress_stack.layout()
            observed.append((
                not self.window.generation_progress.isHidden(),
                not self.window.batch_progress.isHidden(),
                layout.indexOf(self.window.generation_progress),
                layout.indexOf(self.window.batch_progress),
                self.window.batch_progress.value(),
                self.window.batch_progress.format(),
                self.window.generation_progress.height(),
                self.window.batch_progress.height(),
                self.window.synthesis_progress_stack.height(),
                layout.spacing(),
            ))
            return "rendered"

        with mock.patch.object(
                self.window, "_generate_current", side_effect=generate):
            self.window._generate_for_sentence_mode()

        self.assertEqual(
            observed, [(
                True, True, 0, 1, 1, "Generate 1 / 3",
                self.window._synthesis_progress_height // 2,
                self.window._synthesis_progress_height // 2,
                self.window._synthesis_progress_height, 0,
            )])
        self.assertTrue(self.window.generation_progress.isHidden())
        self.assertFalse(self.window.batch_progress.isHidden())
        self.assertFalse(self.window.synthesis_progress_stack.isHidden())
        self.assertEqual(
            self.window.batch_progress.height(),
            self.window._synthesis_progress_height)
        self.window._end_sentence_batch()
        self.assertTrue(self.window.batch_progress.isHidden())
        self.assertTrue(self.window.synthesis_progress_stack.isHidden())

    def test_generation_progress_cleans_up_after_failure(self):
        with mock.patch.object(
                self.window, "_generate_current",
                side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                self.window._generate_for_sentence_mode()

        self.assertTrue(self.window.generation_progress.isHidden())
        self.assertTrue(self.window.synthesis_progress_stack.isHidden())

    def test_rerender_uses_current_generation_progress(self):
        observed = []

        def rerender(**_kwargs):
            observed.append((
                not self.window.generation_progress.isHidden(),
                self.window.generation_progress.format(),
            ))
            return "rendered"

        with mock.patch.object(
                self.window, "_rerender_current", side_effect=rerender):
            result = self.window.on_rerender()

        self.assertEqual(result, "rendered")
        self.assertEqual(
            observed, [(True, "Re-rendering current sentence...")])
        self.assertTrue(self.window.generation_progress.isHidden())
        self.assertTrue(self.window.synthesis_progress_stack.isHidden())

    def test_generated_voice_inline_q_uses_declared_phone_inventory(self):
        backend = mock.Mock()
        backend.voice_metadata.return_value = {
            "phones": ["aa", "q", "pau"],
        }

        expanded, extra, dropped = self.window._prepare_inline_phones(
            "an [q] onset", "en", "lem", backend=backend)

        self.assertIn("qphon0x", expanded)
        self.assertEqual(extra, {"qphon0x": ["q"]})
        self.assertEqual(dropped, set())

    def test_builtin_english_inline_q_keeps_kal_compatibility_filter(self):
        backend = mock.Mock()
        backend.voice_metadata.return_value = {}

        expanded, extra, dropped = self.window._prepare_inline_phones(
            "an [q] onset", "en", "kal_diphone", backend=backend)

        self.assertNotIn("qphon0x", expanded)
        self.assertEqual(extra, {})
        self.assertEqual(dropped, {"q"})

    def test_synthesis_unlock_uses_current_sentence_control_context(self):
        self.window.mode_tabs.setCurrentIndex(1)
        self.assertFalse(self.window.sidebar_editor.isEnabled())

        self.window._set_synthesis_busy(True)
        self.window.sentences_view.set_selected_indices([0])
        self.assertTrue(self.window.sidebar_editor.isEnabled())
        self.assertFalse(self.window.lang.isEnabled())
        self.window._set_synthesis_busy(False)

        for name in ("engine", "lang", "voicebank", "speed", "speed_val"):
            with self.subTest(control=name):
                self.assertTrue(getattr(self.window, name).isEnabled())
        # Pitch remains intentionally unavailable for the pure-Python engine.
        self.assertFalse(self.window.pitch.isEnabled())
        # Gain is intentionally unavailable until the sentence has audio.
        self.assertFalse(self.window.speech_gain.isEnabled())

        self.window.sentences_view.set_selected_indices([])
        self.assertFalse(self.window.sidebar_editor.isEnabled())
        self.window.sentences_view.set_selected_indices([0])
        self.assertTrue(self.window.lang.isEnabled())

    def test_unavailable_combo_rows_use_availability_delegate(self):
        combo = self.window.parameter_mode
        self.assertIsInstance(
            combo.itemDelegate(), fg.AvailabilityItemDelegate)
        pitch = combo.findData("pitch")
        self.assertGreaterEqual(pitch, 0)
        self.assertFalse(combo.model().item(pitch).isEnabled())
        self.assertNotEqual(
            fg.AvailabilityItemDelegate.DISABLED_BACKGROUND, "#FFFFFF")
        self.assertNotIn("font-style: italic", fg.XP_QSS)

    def test_text_edit_discards_stale_phrase_audio_but_revert_restores_it(self):
        state = self.window.sentences[0]
        state["text"] = "old sentence"
        state["rendered_text"] = "old sentence"
        state["phrases"] = [{"id": "old", "text": "old sentence"}]
        preview = np.ones(20, np.float32)
        state["phrase_previews"] = {"old": (preview, 16000)}

        self.assertTrue(self.window._apply_sentence_text_edit(
            0, "new sentence"))
        self.assertEqual(state["phrase_previews"], {})
        self.assertTrue(self.window._apply_sentence_text_edit(
            0, "old sentence"))
        self.assertIn("old", state["phrase_previews"])
        self.assertIs(state["phrase_previews"]["old"][0], preview)

    def test_structural_undo_snapshots_share_immutable_pcm(self):
        syn = fc.Synthesis(
            np.linspace(-.2, .2, 200, dtype=np.float32), 1000,
            [fc.Segment("a", 0, .1), fc.Segment("b", .1, .2)])
        self.window.waveform.set_synthesis(syn)

        before = self.window.waveform.structure_snapshot()

        self.assertIs(before["chunks"][0], self.window.waveform._chunks[0])
        self.assertIs(before["base_audio"][0],
                      self.window.waveform.base_audio[0])
        expected = before["chunks"][0].copy()
        self.window.waveform.set_factor(0, 1.5)
        self.assertTrue(self.window.waveform.restore_structure(before))
        np.testing.assert_array_equal(
            self.window.waveform._chunks[0], expected)

    def test_vocal_tract_curve_defaults_and_range_switching(self):
        track = self.window.vocal_tract_track
        track.set_data([(0.0, 1.0)], ["e"], [], [])
        self.assertFalse(track.chipmunk_range())
        self.assertEqual(track.targets()[0], (0.0, 1.0))
        self.assertEqual(track.targets()[-1], (1.0, 1.0))
        self.assertTrue(all(value == 1.0 for _time, value in track.targets()))
        self.window.vocal_tract_chipmunk.setChecked(True)
        track.set_uniform_ratio(track.profile.expanded_min_ratio, emit=False)
        self.assertTrue(track.chipmunk_range())
        self.assertAlmostEqual(
            track.ratio_range()[0], track.profile.expanded_min_ratio,
            places=3)
        self.window.vocal_tract_chipmunk.setChecked(False)
        self.app.processEvents()
        self.assertFalse(track.chipmunk_range())
        self.assertAlmostEqual(
            track.ratio_range()[0], track.profile.realistic_min_ratio,
            places=3)
        self.assertLessEqual(track.profile.expanded_min_ratio, 0.70)

    def test_vocal_tract_in_range_toggle_does_not_change_sound_value(self):
        track = self.window.vocal_tract_track
        track.set_data([(0.0, 1.0)], ["e"], [], [])
        track.set_uniform_ratio(1.02, emit=False)
        before = track.targets()
        self.window.vocal_tract_chipmunk.setChecked(True)
        self.app.processEvents()
        self.assertEqual(track.targets(), before)
        self.window.vocal_tract_chipmunk.setChecked(False)
        self.app.processEvents()
        self.assertEqual(track.targets(), before)

    def test_vocal_tract_state_is_sentence_local(self):
        self.window.vocal_tract_value.setValue(0.95)
        self.window._set_uniform_vocal_tract_ratio()
        self.window._capture_active_sentence()
        self.assertAlmostEqual(
            self.window.sentences[0]["vocal_tract_length_ratio"],
            0.95, places=6)
        second = self.window._new_sentence_state("second")
        second["vocal_tract_length_ratio"] = 1.04
        second["chipmunk_range"] = False
        self.window.sentences.append(second)
        self.window._active_sentence_index = 1
        self.window._restore_sentence(1)
        self.assertAlmostEqual(self.window.vocal_tract_value.value(), 1.04,
                               places=3)

    def test_vocal_tract_curve_edit_is_pending_until_it_matches_render(self):
        segments = [fc.Segment("e", 0.0, 0.24)]
        phase = np.arange(3840, dtype=np.float64) / 16000.0
        samples = (0.2 * np.sin(2 * np.pi * 180.0 * phase)).astype(
            np.float32)
        syn = self.window._apply_output_faults(
            fc.Synthesis(samples, 16000, segments, phones=["e"]),
            vocal_tract_ratio=1.0,
        )
        self.window._show_synthesis(syn)
        self.window.sentences[0]["rendered"] = True
        self.window._commit_rendered_state(syn)
        self.window.vocal_tract_track.set_uniform_ratio(0.95, emit=True)
        self.assertEqual(self.window.current.vocal_tract_mode, "curve")
        self.assertTrue(self.window.sentences[0]["needs_rerender"])
        self.window.vocal_tract_track.clear_override()
        self.assertFalse(self.window.sentences[0]["needs_rerender"])

    def test_legacy_vocal_tract_scalar_loads_as_uniform_curve(self):
        row = {
            "text": "e", "phones": ["e"],
            "segments": [fc.Segment("e", 0.0, 0.2)],
            "vocal_tract_length_ratio": 0.92,
            "vocal_tract_requested_ratio": 0.92,
            "applied_vocal_tract_length_ratio": 0.92,
            "needs_rerender": False,
        }
        state = self.window._state_from_project_row(row)
        syn = state["synthesis"]
        self.assertEqual(syn.vocal_tract_mode, "curve")
        self.assertEqual(syn.vocal_tract_override,
                         [(0.0, 0.92), (0.2, 0.92)])

    def test_speech_and_sentence_views_and_arrow_combos_exist(self):
        self.assertEqual(self.window.mode_tabs.count(), 2)
        self.assertEqual([self.window.mode_tabs.tabText(index)
                          for index in range(2)],
                         ["Speech", "Sentences"])
        self.assertFalse(hasattr(self.window, "song_view"))
        self.assertFalse(hasattr(self.window, "song_toggle"))
        for combo in (self.window.engine, self.window.lang,
                      self.window.sentence_select, self.window.input_mode,
                      self.window.parameter_mode):
            self.assertIsInstance(combo, fg.ArrowComboBox)
        self.assertIsInstance(
            self.window.pitch_scroll, fg.QtWidgets.QScrollBar)
        self.assertEqual(self.window.pitch_scroll.orientation(), fg.Qt.Vertical)
        self.assertEqual(self.window.pitch_zoom_in.text(), "+")
        self.assertEqual(self.window.pitch_zoom_out.text(), "-")
        self.assertEqual(self.window.pitch_navigator.width(), 24)
        self.assertTrue(self.window.follow_playhead.isCheckable())
        self.assertTrue(self.window.follow_playhead.isChecked())
        self.assertTrue(
            self.window.sentences_view.follow_spoken_sentence.isChecked())
        self.assertEqual(self.window.timing_consonants.text(),
                         "Consonant velocity")
        self.assertEqual(self.window.timing_vowels.text(), "Vowel length")
        self.assertEqual(self.window.action_open.text().split("\t", 1)[0],
                         "Open Project JSON...")
        self.assertEqual(self.window.sentences_view.play_all.text(), "Play all")

    def test_sentence_selection_changes_play_all_label(self):
        view = self.window.sentences_view
        view.refresh(self.window.sentences)
        view.set_selected_indices([0])
        self.assertEqual(view.play_all.text(), "Play selected")
        view.set_selected_indices([])
        self.assertEqual(view.play_all.text(), "Play all")
        self.assertFalse(hasattr(self.window, "action_open_legacy"))
        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        self.window._update_fault_availability()
        self.assertFalse(self.window.pitch.isHidden())
        self.assertFalse(self.window.fall.isHidden())

    def test_sentence_row_selection_defers_hidden_editor_hydration(self):
        first_audio = np.full(2000, 0.1, np.float32)
        second_audio = np.full(3000, 0.2, np.float32)
        first_syn = fc.Synthesis(
            first_audio, 1000, [fc.Segment("a", 0.0, 2.0)],
            phones=["a"], text="first")
        second_syn = fc.Synthesis(
            second_audio, 1000, [fc.Segment("b", 0.0, 3.0)],
            phones=["b"], text="second")
        first = self.window._new_sentence_state("first")
        first.update({
            "synthesis": first_syn,
            "editor_segments": copy.deepcopy(first_syn.segments),
            "preview_audio": first_audio,
            "preview_sr": 1000,
            "rendered": True,
        })
        second = self.window._new_sentence_state("second")
        second.update({
            "synthesis": second_syn,
            "editor_segments": copy.deepcopy(second_syn.segments),
            "preview_audio": second_audio,
            "preview_sr": 1000,
            "rendered": True,
        })
        self.window.sentences = [first, second]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)
        self.window.mode_tabs.setCurrentIndex(1)
        self.app.processEvents()

        with mock.patch.object(
                self.window.waveform, "set_synthesis",
                wraps=self.window.waveform.set_synthesis) as hydrate, \
                mock.patch.object(
                    self.window, "_refresh_voicebanks",
                    wraps=self.window._refresh_voicebanks) as refresh_voices:
            self.window._on_sentences_selection_changed([1])
            self.assertEqual(hydrate.call_count, 0)
            self.assertEqual(refresh_voices.call_count, 0)
            self.assertEqual(self.window._active_sentence_index, 1)
            self.assertIs(self.window.current, second_syn)
            self.assertIs(self.window._editor_sentence_state, first)
            self.assertEqual(self.window.waveform.segments[0].phone, "a")

            self.window.mode_tabs.setCurrentIndex(0)
            self.app.processEvents()
            self.assertEqual(hydrate.call_count, 1)
            self.assertIs(self.window._editor_sentence_state, second)
            self.assertEqual(self.window.waveform.segments[0].phone, "b")

    def test_sentence_restore_hydrates_once_and_shares_immutable_pcm(self):
        samples = np.linspace(-0.2, 0.2, 4000, dtype=np.float32)
        syn = fc.Synthesis(
            samples, 1000,
            [fc.Segment("a", 0.0, 2.0),
             fc.Segment("b", 2.0, 4.0)],
            phones=["a", "b"], text="two phones")
        state = self.window._new_sentence_state("two phones")
        state.update({
            "synthesis": syn,
            "editor_segments": copy.deepcopy(syn.segments),
            "timing_factors": [1.0, 1.0],
            "preview_audio": samples,
            "preview_sr": 1000,
            "rendered": True,
        })
        self.window.sentences = [state]
        self.window._active_sentence_index = 0

        with mock.patch.object(
                self.window.waveform, "set_synthesis",
                wraps=self.window.waveform.set_synthesis) as hydrate:
            self.window._restore_sentence(0)

        self.assertEqual(hydrate.call_count, 1)
        self.assertTrue(np.shares_memory(
            self.window.waveform.audio, samples))
        self.assertTrue(np.shares_memory(
            self.window.waveform.base_audio[0], samples))
        self.assertTrue(np.shares_memory(
            self.window.waveform._chunks[1], samples))

    def test_voicebank_list_collapses_resizes_and_reserves_scrollbar_space(self):
        self.assertTrue(self.window.voicebank_heading.isChecked())
        self.assertFalse(self.window.voicebank.isHidden())
        self.assertGreaterEqual(
            self.window.sidebar_editor.layout().contentsMargins().right(), 14)
        self.assertGreaterEqual(self.window.sidebar.width(), 240)
        watched = {os.path.normcase(os.path.abspath(path)) for path in
                   self.window._voice_root_watcher.directories()}
        self.assertIn(os.path.normcase(os.path.abspath(
            self.window.fest.generated_voice_root())), watched)

        self.window.voicebank_heading.setChecked(False)
        self.assertTrue(self.window.voicebank.isHidden())
        self.assertTrue(self.window.voicebank_resize.isHidden())
        self.assertEqual(self.window.voicebank_heading.arrowType(),
                         fg.Qt.RightArrow)

        self.window.voicebank_heading.setChecked(True)
        self.window.voicebank.setFixedHeight(143)
        self.window.voicebank_resize.resized.emit(143)
        self.assertFalse(self.window.voicebank.isHidden())
        self.assertEqual(self.window.cfg["voicebank_list_height"], 143)

    def test_builtin_kal_stays_visible_when_its_windows_mirror_is_stale(self):
        self.window.fest = fc.FestivalWSLBackend({"festival_wsl": {
            "voices": {"kal_diphone": {
                "dir": r"Z:\missing\kal_diphone",
                "voice": "voice_broken_kal",
            }},
        }})
        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        self.window.lang.blockSignals(True)
        self.window.lang.setCurrentText("Japanese")
        self.window.lang.blockSignals(False)
        self.window.sentences[0].update({
            "language": "Japanese", "lang_code": "ja",
            "voicebank": "",
        })

        self.window._refresh_voicebanks()
        self.app.processEvents()

        kal_items = [
            self.window.voicebank.item(row)
            for row in range(self.window.voicebank.count())
            if self.window.voicebank.item(row).data(fg.Qt.UserRole)
            == "kal_diphone"
        ]
        self.assertEqual(len(kal_items), 1)
        self.assertIs(self.window.voicebank.currentItem(), kal_items[0])
        self.assertEqual(kal_items[0].text(), "kal_diphone")
        self.assertIn(
            "/usr/share/festival/voices/english/kal_diphone",
            kal_items[0].toolTip(),
        )
        self.assertEqual(self.window.lang.currentText(), "English")
        self.assertEqual(self.window.sentences[0]["voicebank"], "kal_diphone")
        self.assertEqual(self.window.sentences[0]["lang_code"], "en")
        self.assertEqual(
            self.window._pending_action(self.window.sentences[0]),
            "generate",
        )

    def test_render_refresh_preserves_unchanged_voice_caches(self):
        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        self.window._variant_cache[("festival_wsl", "lem")] = {
            "t-eh": [{"id": "old"}]}
        self.window.fest._alternatives["lem"] = {
            "t-eh": [{"id": "old"}]}
        self.window.fest._sustains[("lem", "eh")] = (
            np.zeros(8, np.float32), 16000)
        self.window.fest._voice_metadata["lem"] = {"language": "ja"}

        self.window._refresh_voice_metadata()

        self.assertIn(("festival_wsl", "lem"), self.window._variant_cache)
        self.assertIn("lem", self.window.fest._alternatives)
        self.assertIn(("lem", "eh"), self.window.fest._sustains)
        self.assertIn("lem", self.window.fest._voice_metadata)

    def test_application_cache_menu_clears_memory_only(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source_voice.wav"
            project = Path(folder) / "project.json"
            exported = Path(folder) / "export.wav"
            for path in (source, project, exported):
                path.write_bytes(b"owned-by-user")
            synthesis = fc.Synthesis(
                np.zeros(160, np.float32), 16000,
                [fc.Segment("a", 0.0, 0.01)])
            self.window.current = synthesis
            self.window.sentences[0]["cache_wav"] = str(project)
            self.window.fest._voice_metadata["fixture"] = {"language": "ja"}
            self.window.fest._alternatives["fixture"] = {"a-b": []}
            self.window.fest._sustains[("fixture", "a")] = (
                np.ones(16, np.float32), 16000)
            self.window._variant_cache[
                ("festival_wsl", "fixture", "token")] = {"a-b": []}

            self.window._refresh_cache_menu()
            self.assertIn("In-memory cache", self.window.cache_usage_action.text())
            with mock.patch("os.remove", side_effect=AssertionError(
                    "cache clearing attempted filesystem deletion")), \
                    mock.patch("shutil.rmtree", side_effect=AssertionError(
                        "cache clearing attempted directory deletion")):
                self.window._clear_application_caches("all")

            self.assertIs(self.window.current, synthesis)
            self.assertEqual(self.window.sentences[0]["cache_wav"],
                             str(project))
            self.assertTrue(all(path.is_file()
                                for path in (source, project, exported)))
            self.assertEqual(self.window._variant_cache, {})
            self.assertEqual(self.window.fest._voice_metadata, {})
            self.assertEqual(self.window.fest._alternatives, {})
            self.assertEqual(self.window.fest._sustains, {})
            self.assertIn("were not touched",
                          self.window.statusBar().currentMessage())

    def test_wsl_settings_exposes_windows_and_wsl_scan_roots(self):
        captured = {}

        def inspect_dialog(dialog):
            dialog.adjustSize()
            dialog.show()
            self.app.processEvents()
            captured["root"] = dialog.findChild(
                fg.QtWidgets.QLineEdit, "generatedVoiceRoot"
            ).text()
            captured["runtime"] = dialog.findChild(
                fg.QtWidgets.QLabel, "generatedVoiceRuntimePath"
            ).text()
            captured["wsl_root"] = dialog.findChild(
                fg.QtWidgets.QLineEdit, "generatedVoiceWSLRoot"
            ).text()
            captured["labels"] = [
                label.text() for label in dialog.findChildren(
                    fg.QtWidgets.QLabel
                )
            ]
            captured["size"] = dialog.size()
            return fg.QtWidgets.QDialog.Rejected

        with mock.patch.object(
            fg.QtWidgets.QDialog, "exec_", inspect_dialog
        ):
            self.window.on_wsl_settings()

        self.assertTrue(captured["root"].endswith("generated_voices"))
        self.assertTrue(captured["runtime"].startswith("/mnt/"))
        self.assertEqual(captured["wsl_root"], "")
        self.assertTrue(any(
            "both scan roots" in text for text in captured["labels"]
        ))
        self.assertLessEqual(captured["size"].width(), 760)
        self.assertLessEqual(captured["size"].height(), 680)

    def test_compact_window_uses_an_independently_scrolling_sidebar(self):
        self.window.resize(1024, 680)
        self.window.show()
        self.app.processEvents()

        self.assertEqual(self.window.size().height(), 680)
        self.assertGreater(
            self.window.sidebar.verticalScrollBar().maximum(), 0)
        self.assertGreaterEqual(
            self.window.sidebar_editor.layout().contentsMargins().right(),
            self.window.sidebar.verticalScrollBar().sizeHint().width() - 2)

    def test_pitch_and_blocks_accept_a_synthesis(self):
        segments = [fc.Segment("pau", 0.0, 0.1),
                    fc.Segment("a", 0.1, 0.3),
                    fc.Segment("pau", 0.3, 0.4),
                    fc.Segment("pau", 0.4, 0.7),
                    fc.Segment("b", 0.7, 0.9),
                    fc.Segment("pau", 0.9, 1.0)]
        syn = fc.Synthesis(np.zeros(16000, np.float32), 16000, segments,
                           text="one? two.", targets=[(0.15, 170), (0.85, 210)],
                           generated_targets=[(0.15, 170), (0.85, 210)])

        self.window.text.setText("one? two.")
        self.window._show_synthesis(syn)

        times = [round(t, 2) for t, _f in self.window.pitch_track.targets()]
        self.assertEqual(times, [0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4,
                                 0.55, 0.7, 0.8, 0.9, 0.95, 1.0])
        self.assertEqual([b["kind"] for b in self.window.intonation.blocks()],
                         ["?", "."])

        blocks = self.window.intonation.blocks()
        self.window._on_intonation_commit(blocks)
        canonical_ground = fc.anchor_phrase_targets(
            [(segment.phone, segment.dur) for segment in segments],
            syn.generated_targets, self.window.pitch.value())
        expected = fc.overlay_intonation_targets(
            canonical_ground, blocks, self.window.pitch.value(),
            self.window.fall.value())
        expected_times = [time for time, _value in expected]
        expected_values = [value for _time, value in expected]
        for time, value in self.window.pitch_track.targets():
            self.assertAlmostEqual(
                value, float(np.interp(time, expected_times,
                                       expected_values)))

    def test_english_pitch_curve_recovers_from_rendered_pitchmarks(self):
        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        segments = [
            fc.Segment("pau", 0.0, 0.08),
            fc.Segment("hh", 0.08, 0.15),
            fc.Segment("eh", 0.15, 0.32),
            fc.Segment("pau", 0.32, 0.40),
        ]
        marks = [
            0.085, 0.091, 0.097, 0.104, 0.111, 0.119, 0.127,
            0.136, 0.145, 0.155, 0.165, 0.176, 0.187, 0.199,
            0.211, 0.224, 0.237, 0.251, 0.265, 0.280, 0.295,
        ]
        synthesis = fc.Synthesis(
            np.zeros(400, np.float32), 1000, segments,
            text="hello", lang="en", voicebank="fixture",
            targets=[], generated_targets=[], target_pitchmarks=marks)

        self.window._show_synthesis(synthesis)

        ground = list(self.window.pitch_track._ground)
        self.assertGreater(len(ground), 3)
        self.assertGreater(max(value for _time, value in ground),
                           min(value for _time, value in ground))
        self.assertTrue(self.window.pitch_track.targets())
        self.assertTrue(all(
            fc.PITCH_MIN_HZ <= value <= fc.PITCH_MAX_HZ
            for _time, value in ground))

    def test_recordings_and_sentence_switching_are_visible(self):
        segments = [fc.Segment("s", 0, .1), fc.Segment("t", .1, .2),
                    fc.Segment("eh", .2, .35)]
        inventory = {"t-eh": [
            {"id": "base", "left_name": "t", "left_context": "z",
             "right_context": "r"},
            {"id": "take1", "left_name": "t__u1", "left_context": "s",
             "right_context": "l"},
        ]}
        self.window.recordings.set_data(
            segments, inventory, {1: "t__u1"}, {})

        self.assertEqual(len(self.window.recordings._rows), 2)
        self.assertTrue(self.window.recordings._rows[0]["inspect_only"])
        self.assertEqual(self.window.recordings._rows[0]["pair"], "s-t")
        self.assertFalse(self.window.recordings._rows[1]["inspect_only"])
        self.assertEqual(self.window.recordings._rows[1]["label"], "Auto take1")
        self.window.text.setText("first sentence")
        self.window._on_sentence_text_edited("first sentence")
        self.window.on_add_sentence()
        self.assertEqual(len(self.window.sentences), 2)
        self.window.text.setText("second sentence")
        self.window._on_sentence_text_edited("second sentence")
        self.window.sentence_select.setCurrentIndex(0)
        self.assertEqual(self.window.text.text(), "first sentence")

    def test_legacy_joins_fault_is_sentence_local_across_switches(self):
        first = self._rendered_sentence_with_unit_choices("first")
        second = self._rendered_sentence_with_unit_choices("second")
        first["fault_mode"]["legacy_joins"] = False
        second["fault_mode"]["legacy_joins"] = False
        expected_selected = {0: "a__context", 1: "b__manual"}
        expected_overrides = {1: "b__manual"}
        self.window.sentences = [first, second]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)
        legacy = self.window.fault_actions["legacy_joins"]

        self.assertFalse(legacy.isChecked())
        legacy.setChecked(True)
        self.app.processEvents()

        self.assertTrue(first["fault_mode"]["legacy_joins"])
        self.assertFalse(second["fault_mode"]["legacy_joins"])
        self.assertEqual(first["synthesis"].selected_units,
                         expected_selected)
        self.assertEqual(first["synthesis"].unit_overrides,
                         expected_overrides)

        self.window.sentence_select.setCurrentIndex(1)
        self.app.processEvents()

        self.assertFalse(legacy.isChecked())
        self.assertTrue(first["fault_mode"]["legacy_joins"])
        self.assertFalse(second["fault_mode"]["legacy_joins"])
        self.assertEqual(second["synthesis"].selected_units,
                         expected_selected)
        self.assertEqual(second["synthesis"].unit_overrides,
                         expected_overrides)

        self.window.sentence_select.setCurrentIndex(0)
        self.app.processEvents()

        self.assertTrue(legacy.isChecked())
        self.assertEqual(self.window.current.selected_units,
                         expected_selected)
        self.assertEqual(self.window.current.unit_overrides,
                         expected_overrides)

    def test_legacy_joins_fault_and_unit_choices_survive_project_reload(self):
        first = self._rendered_sentence_with_unit_choices("first")
        second = self._rendered_sentence_with_unit_choices("second")
        first["fault_mode"]["legacy_joins"] = True
        second["fault_mode"]["legacy_joins"] = False
        self.window.sentences = [first, second]
        self.window._active_sentence_index = 1
        self.window._refresh_sentence_selector(1)
        self.window._restore_sentence(1)

        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "Legacy Join State"
            self.assertTrue(self.window.on_save_project(project))
            self.assertTrue(
                self.window.on_open_project(project / "project.json"))

        self.assertEqual(self.window._active_sentence_index, 1)
        self.assertFalse(
            self.window.fault_actions["legacy_joins"].isChecked())
        self.assertTrue(
            self.window.sentences[0]["fault_mode"]["legacy_joins"])
        self.assertFalse(
            self.window.sentences[1]["fault_mode"]["legacy_joins"])
        for state in self.window.sentences:
            self.assertEqual(state["synthesis"].selected_units,
                             {0: "a__context", 1: "b__manual"})
            self.assertEqual(state["synthesis"].unit_overrides,
                             {1: "b__manual"})

        self.window.sentence_select.setCurrentIndex(0)
        self.app.processEvents()

        self.assertTrue(
            self.window.fault_actions["legacy_joins"].isChecked())
        self.assertEqual(self.window.current.selected_units,
                         {0: "a__context", 1: "b__manual"})
        self.assertEqual(self.window.current.unit_overrides,
                         {1: "b__manual"})

    def test_recording_inspector_explains_oto_only_sibilant_choice(self):
        segments = [fc.Segment("s", 0, .1), fc.Segment("ay", .1, .3),
                    fc.Segment("z", .3, .4), fc.Segment("er", .4, .6)]
        choices = [
            {"id": "base", "left_name": "ay", "index_name": "ay-z",
             "left_context": "ch", "right_context": "d",
             "alias": "ay z", "wav": "filename_says_vowel.wav",
             "oto_line": 10},
            {"id": "take1", "left_name": "ay__u1",
             "index_name": "ay__u1-z", "left_context": "r",
             "right_context": "*", "alias": "ay z1",
             "wav": "opaque.wav", "oto_line": 20},
            {"id": "take2", "left_name": "ay__u2",
             "index_name": "ay__u2-z", "left_context": "r",
             "right_context": "oy", "alias": "ay z2",
             "right_context_source": "adjacent_oto_edge",
             "wav": "filename_says_stop.wav", "oto_line": 30},
        ]
        syn = fc.Synthesis(np.zeros(600, np.float32), 1000, segments,
                           selected_units={1: "ay__u2"})
        self.window.current = syn
        self.window.waveform.set_synthesis(syn)
        row = {"index": 1, "pair": "ay-z", "choices": choices,
               "actual": "ay__u2", "manual": None,
               "choice": choices[2]}

        with mock.patch.object(
                fg.QtWidgets.QMessageBox, "information") as information:
            self.window._show_recording_details(row)

        message = information.call_args.args[2]
        self.assertIn("verified vowel right context", message)
        self.assertIn("Right OTO evidence: oy: known vowel", message)
        self.assertIn("recovered from the immediately adjacent ordered OTO",
                      message)
        self.assertIn("WAV filenames are ignored", message)

    def test_small_control_regressions(self):
        self.window.fall.setValue(16)

        self.assertEqual(self.window.fall.value(), 16)
        self.assertNotIn("QComboBox::down-arrow", fg.XP_QSS)
        self.assertTrue(issubclass(fg.ArrowProxyStyle,
                                   fg.QtWidgets.QProxyStyle))
        self.assertFalse(
            self.window.waveform.plot.getViewBox().autoRangeEnabled()[1])
        self.window.text.setFocus()
        self.assertTrue(self.window._shortcut_blocked())
        self.assertTrue(hasattr(self.window, "output_gain_slider"))
        self.assertIsInstance(self.window.output_gain_slider, fg.ResetSlider)
        self.assertEqual(self.window.speaker_portrait.width(),
                         self.window.speaker_portrait.height())
        self.assertFalse(self.window.waveform.playhead.movable)
        self.window.waveform.sr = 16000
        self.window.waveform.audio = np.zeros(32000, np.float32)
        self.window.waveform.timeline.set_duration(2.0)
        self.window.waveform.timeline.timeChanged.emit(1.25)
        self.assertAlmostEqual(self.window.waveform.playhead_time(), 1.25)
        self.assertTrue(
            self.window.sidebar.isAncestorOf(self.window.output_gain_slider))

    def test_speaker_portrait_tracks_restored_sentence_voice(self):
        portrait = Path(self._config_tmp.name) / "voice-a.png"
        pixmap = fg.QtGui.QPixmap(12, 12)
        pixmap.fill(fg.QtGui.QColor("#D03030"))
        self.assertTrue(pixmap.save(str(portrait)))
        self.window.cfg["speaker_portrait"] = str(portrait)
        self.window.cfg["voice_portraits"] = {
            "diphone|voice_a": str(portrait),
        }

        class FakeVoices:
            @staticmethod
            def voicebanks():
                return [
                    {"name": "voice_a", "dir": "a", "ok": True,
                     "source": "test"},
                    {"name": "voice_b", "dir": "b", "ok": True,
                     "source": "test"},
                ]

            @staticmethod
            def default_voicebank():
                return "voice_a"

        self.window.backend = FakeVoices()
        first = self.window._new_sentence_state("first")
        first.update({"engine": "diphone", "voicebank": "voice_a"})
        second = self.window._new_sentence_state("second")
        second.update({"engine": "diphone", "voicebank": "voice_b"})
        self.window.sentences = [first, second]

        self.window._active_sentence_index = 0
        self.window._restore_sentence(0)
        first_key = self.window.speaker_portrait.pixmap().cacheKey()
        self.assertEqual(self.window.speaker_portrait.portrait_path,
                         str(portrait))
        self.assertEqual(self.window.speaker_portrait.speaker_name, "voice_a")

        self.window._active_sentence_index = 1
        self.window._restore_sentence(1)

        self.assertEqual(self.window._current_voicebank(), "voice_b")
        self.assertEqual(self.window.speaker_portrait.portrait_path, "")
        self.assertEqual(self.window.speaker_portrait.speaker_name, "voice_b")
        self.assertTrue(self.window.speaker_portrait.fallback_color)
        self.assertIn("voice_b", self.window.speaker_portrait.toolTip())
        self.assertNotEqual(self.window.speaker_portrait.pixmap().cacheKey(),
                            first_key)

    def test_playhead_ruler_stays_aligned_during_pre_zero_pan(self):
        waveform = self.window.waveform
        waveform.set_workspace_duration(2.0)

        waveform.plot.setXRange(-0.2, 0.8, padding=0)
        self.app.processEvents()

        plot_range = waveform.plot.getViewBox().viewRange()[0]
        ruler_range = waveform.timeline.getViewBox().viewRange()[0]
        self.assertAlmostEqual(plot_range[0], ruler_range[0], places=6)
        self.assertAlmostEqual(plot_range[1], ruler_range[1], places=6)

        waveform.set_playhead(0.35)
        self.assertAlmostEqual(waveform.playhead.value(),
                               waveform.timeline.cursor.value(), places=6)

    def test_playhead_follow_jumps_by_page_and_can_be_disabled(self):
        waveform = self.window.waveform
        waveform.set_workspace_duration(4.0)
        waveform.plot.setXRange(0.0, 1.0, padding=0)
        self.app.processEvents()

        self.assertTrue(self.window._follow_playhead_if_needed(1.2))
        followed = waveform.plot.getViewBox().viewRange()[0]
        self.assertLess(followed[0], 1.2)
        self.assertGreater(followed[1], 1.2)

        self.window.follow_playhead.setChecked(False)
        waveform.plot.setXRange(0.0, 1.0, padding=0)
        self.app.processEvents()
        before = list(waveform.plot.getViewBox().viewRange()[0])
        self.assertFalse(self.window._follow_playhead_if_needed(1.2))
        self.assertEqual(before, waveform.plot.getViewBox().viewRange()[0])

    def test_rerender_display_preserves_exact_view_and_playhead(self):
        segments = [
            fc.Segment("a", 0.0, 1.0),
            fc.Segment("r", 1.0, 2.0),
        ]
        first = fc.Synthesis(
            np.zeros(2000, np.float32), 1000, segments,
            phones=["a", "r"])
        replacement = fc.Synthesis(
            np.ones(2000, np.float32) * .02, 1000, segments,
            phones=["a", "r"])
        self.window._show_synthesis(first)
        waveform = self.window.waveform
        waveform.plot.getViewBox().setXRange(.45, 1.05, padding=0)
        waveform.set_playhead(.72)
        self.app.processEvents()
        before = tuple(waveform.plot.getViewBox().viewRange()[0])

        self.window._show_synthesis(
            replacement, preserve_view=True, focus_timeline=False)
        self.app.processEvents()
        after = waveform.plot.getViewBox().viewRange()[0]

        self.assertAlmostEqual(after[0], before[0], places=6)
        self.assertAlmostEqual(after[1], before[1], places=6)
        self.assertAlmostEqual(waveform.playhead_time(), .72, places=6)

    def test_long_waveform_uses_cached_viewport_peak_columns(self):
        samples = np.zeros(1_000_000, np.float32)
        samples[500_000] = .92
        samples[500_001] = -.81
        cache = fg.WaveformPeakCache(samples)

        x, y = cache.display(0.0, 62.5, 16000, 800)

        self.assertLessEqual(len(x), 800 * 2 + 1)
        self.assertEqual(len(x), len(y))
        self.assertEqual(cache.last_mode, "envelope")
        self.assertLessEqual(cache.last_source_count, 800 * 2 + 2)
        self.assertLessEqual(cache.last_block_size, 62.5 * 16000 / 800)
        self.assertGreater(
            cache.last_block_size * fg.WAVEFORM_SUMMARY_GROWTH,
            62.5 * 16000 / 800)
        self.assertAlmostEqual(float(np.nanmax(y)), .92, places=5)
        self.assertAlmostEqual(float(np.nanmin(y)), -.81, places=5)

        coarse_x, coarse_y = cache.display(0.0, 62.5, 16000, 10)
        self.assertLessEqual(len(coarse_x), 21)
        self.assertLessEqual(cache.last_source_count, 22)
        self.assertGreater(cache.last_block_size, 1024)
        self.assertAlmostEqual(float(np.nanmax(coarse_y)), .92, places=5)
        self.assertAlmostEqual(float(np.nanmin(coarse_y)), -.81, places=5)

        useful_x, useful_y = cache.display(0.0, 0.08, 44100, 1600)
        self.assertEqual(cache.last_mode, "line")
        self.assertEqual(cache.last_block_size, 1)
        self.assertEqual(len(useful_x), len(useful_y))
        self.assertLessEqual(len(useful_x), 1600 * 2 + 2)
        self.assertFalse(np.isnan(useful_y).any())
        self.assertTrue(np.all(np.diff(useful_x) >= 0.0))

        raw_x, raw_y = cache.display(31.25, 31.251, 16000, 800)
        self.assertEqual(cache.last_block_size, 1)
        self.assertEqual(len(raw_x), len(raw_y))
        self.assertFalse(np.isnan(raw_y).any())

        waveform = self.window.waveform
        waveform.sr = 16000
        waveform.audio = samples
        waveform.set_workspace_duration(62.5)
        waveform.plot.setXRange(0.0, 62.5, padding=0)
        waveform._redraw()
        self.app.processEvents()
        data_width = max(32, int(round(
            waveform.plot.getViewBox().sceneBoundingRect().width())))
        self.assertLessEqual(len(waveform.curve.xData), data_width * 3)

    def test_waveform_summary_work_is_bounded_across_middle_zoom_levels(self):
        samples = np.sin(
            np.arange(2_000_000, dtype=np.float32) * .037)
        cache = fg.WaveformPeakCache(samples)
        sr, width = 16000, 1000

        for seconds in (2.0, 8.0, 30.0, 90.0, 125.0):
            x, y = cache.display(0.0, seconds, sr, width)
            self.assertEqual(len(x), len(y))
            self.assertLessEqual(len(x), width * 2 + 2)
            if cache.last_mode == "envelope":
                self.assertLessEqual(cache.last_source_count, width * 2 + 2)
                samples_per_pixel = seconds * sr / width
                self.assertLessEqual(
                    cache.last_block_size, samples_per_pixel)
                self.assertGreater(
                    cache.last_block_size * fg.WAVEFORM_SUMMARY_GROWTH,
                    samples_per_pixel)

    def test_hidden_parameter_tracks_defer_linked_zoom_lod(self):
        self.window.resize(1200, 820)
        self.window.show()
        self.window.parameter_stack.setCurrentIndex(0)
        self.app.processEvents()
        tracks = (
            self.window.timing,
            self.window.pitch_track,
            self.window.voicing_track,
            self.window.vocal_tract_track,
            self.window.intonation,
            self.window.recordings,
        )
        for track in tracks:
            track._lod_timer.stop()

        self.assertTrue(self.window.timing.isVisible())
        self.window.timing._schedule_lod_redraw()
        self.window.pitch_track._schedule_lod_refresh()
        self.window.voicing_track._schedule_lod_refresh()
        self.window.vocal_tract_track._schedule_lod_refresh()
        self.window.intonation._schedule_lod_redraw()
        self.window.recordings._schedule_lod_redraw()
        self.assertTrue(self.window.timing._lod_timer.isActive())
        for track in tracks[1:]:
            self.assertFalse(track._lod_timer.isActive())

        self.window.timing._lod_timer.stop()
        self.window.parameter_stack.setCurrentIndex(1)
        self.app.processEvents()
        self.window.pitch_track._lod_timer.stop()
        self.window.timing._schedule_lod_redraw()
        self.window.pitch_track._schedule_lod_refresh()
        self.assertFalse(self.window.timing._lod_timer.isActive())
        self.assertTrue(self.window.pitch_track._lod_timer.isActive())

    def test_dense_waveform_uses_overview_lod_until_zoomed_in(self):
        self.window.resize(1200, 820)
        self.window.show()
        self.app.processEvents()
        count, step, sr = 600, 0.05, 1000
        segments = [
            fc.Segment("pau" if index % 5 in (0, 1) else "aa",
                       index * step, (index + 1) * step)
            for index in range(count)
        ]
        synthesis = fc.Synthesis(
            np.zeros(int(count * step * sr), np.float32), sr, segments,
            text="dense overview")
        waveform = self.window.waveform

        waveform.set_synthesis(synthesis)
        waveform.plot.setXRange(0.0, count * step, padding=0)
        waveform._refresh_visible_view()
        self.app.processEvents()

        width = max(32.0, waveform.plot.getViewBox()
                    .sceneBoundingRect().width())
        marker_count = len(waveform.boundary_overview.xData) // 3
        self.assertFalse(waveform._boundary_lod_detailed)
        self.assertFalse(waveform._visible_boundary_indices)
        self.assertLessEqual(
            marker_count,
            int(np.ceil(width / fg.BOUNDARY_OVERVIEW_BUCKET_PX)) + 2)
        self.assertGreater(
            float(np.nanmax(waveform.boundary_overview.yData)), -0.7)
        self.assertFalse(waveform.phone_labels)
        self.assertFalse(waveform._visible_field_indices)

        waveform.plot.setXRange(0.0, 0.4, padding=0)
        waveform._refresh_visible_view()
        self.app.processEvents()

        self.assertTrue(waveform._boundary_lod_detailed)
        self.assertTrue(waveform._visible_boundary_indices)
        self.assertTrue(waveform.phone_labels)
        self.assertTrue(waveform._visible_field_indices)

    def test_long_waveform_virtualizes_phone_controls_and_join_overlays(self):
        self.window.resize(1200, 820)
        self.window.show()
        self.app.processEvents()
        count, step, sr = 2000, 0.05, 1000
        segments = [
            fc.Segment("aa", index * step, (index + 1) * step)
            for index in range(count)
        ]
        synthesis = fc.Synthesis(
            np.zeros(int(count * step * sr), np.float32), sr, segments,
            text="virtualized long sentence")
        synthesis.splice_records = [{
            "segment_index": index,
            "time": segment.end,
            "crossover_start": segment.end - 0.01,
            "crossover_end": segment.end + 0.01,
        } for index, segment in enumerate(segments[:-1])]
        waveform = self.window.waveform
        waveform.set_synthesis(synthesis)
        waveform.set_join_overlays(synthesis.splice_records)
        waveform.set_join_overlay_visible(True)
        waveform.plot.setXRange(0.0, count * step, padding=0)
        waveform._refresh_visible_view()
        self.app.processEvents()

        width = max(
            32.0,
            waveform.plot.getViewBox().sceneBoundingRect().width())
        self.assertFalse(waveform._join_overlay_lod_detailed)
        self.assertLessEqual(
            waveform._join_overlay_display_count,
            int(np.ceil(width / fg.JOIN_OVERVIEW_BUCKET_PX)) + 2)
        self.assertEqual(
            sum(field is not None for field in waveform.fields), 0)
        self.assertEqual(
            sum(line is not None for line in waveform.boundaries), 0)

        waveform.plot.setXRange(0.0, 0.4, padding=0)
        waveform._refresh_visible_view()
        self.app.processEvents()
        live_fields = {
            index for index, field in enumerate(waveform.fields)
            if field is not None
        }
        live_boundaries = {
            index for index, line in enumerate(waveform.boundaries)
            if line is not None
        }
        self.assertEqual(live_fields, waveform._visible_field_indices)
        self.assertEqual(live_boundaries,
                         waveform._visible_boundary_indices)
        self.assertLessEqual(len(live_fields), 12)
        self.assertLessEqual(len(live_boundaries), 12)

        waveform.plot.setXRange(50.0, 50.4, padding=0)
        waveform._refresh_visible_view()
        self.app.processEvents()
        self.assertFalse(any(
            line is not None for line in waveform.boundaries[:100]))
        self.assertLessEqual(
            sum(line is not None for line in waveform.boundaries), 12)

    def test_long_linguistic_overlay_uses_bounded_viewport_geometry(self):
        count, step = 2000, 0.05
        duration = count * step
        spans = [
            (index * step, (index + 1) * step)
            for index in range(count)
        ]
        track = fg.PitchTrack()
        try:
            track.resize(960, 180)
            track.show()
            track.setXRange(0.0, duration, padding=0)
            track.set_data(
                spans, ["aa"] * count,
                [(0.0, 160.0), (duration, 160.0)])
            track.set_linguistic_unit_debug({"units": [{
                "index": index,
                "phone_start": index,
                "phone_end": index + 1,
                "phones": ["aa"],
                "display_label": "aa",
                "kind": "syllable",
            } for index in range(count)]})
            self.app.processEvents()

            width = max(
                32.0, track.getViewBox().sceneBoundingRect().width())
            self.assertFalse(track._syllable_lod_detailed)
            self.assertLessEqual(
                track._syllable_display_count,
                int(np.ceil(
                    width / fg.LINGUISTIC_OVERVIEW_BUCKET_PX)) + 2)

            track.setXRange(0.0, 0.4, padding=0)
            track._refresh_syllable_vertical_geometry()
            self.app.processEvents()
            self.assertTrue(track._syllable_lod_detailed)
            self.assertLessEqual(track._syllable_display_count, 10)
        finally:
            track.close()
            track.deleteLater()
            self.app.processEvents()

    def test_dense_parameter_tracks_use_bounded_overviews(self):
        count, step = 600, 0.05
        duration = count * step
        spans = [(index * step, (index + 1) * step)
                 for index in range(count)]
        phones = ["pau" if index % 3 == 0 else "aa"
                  for index in range(count)]
        segments = [fc.Segment(phone, start, end)
                    for phone, (start, end) in zip(phones, spans)]
        ground_times = np.linspace(0.0, duration, 5001)
        ground_values = 160.0 + 18.0 * np.sin(ground_times * 0.7)
        generated = list(zip(ground_times, ground_values))
        blocks = [
            {"start": start, "end": end,
             "kind": ("?", "!", ",", ".")[index % 4]}
            for index, (start, end) in enumerate(spans)
        ]
        pitch = fg.PitchTrack()
        timing = fg.TimingTrack()
        recordings = fg.RecordingTrack()
        intonation = fg.IntonationTrack()
        tracks = [pitch, timing, recordings, intonation]
        try:
            for track in tracks:
                track.resize(960, 180)
                track.show()
                track.setXRange(0.0, duration, padding=0)
            self.app.processEvents()

            pitch.set_data(spans, phones, generated)
            timing.set_segments(spans, [1.0] * count, phones)
            recordings.set_data(segments, {}, {}, {})
            intonation.set_blocks(blocks)
            self.app.processEvents()

            expected = np.interp(pitch._times, ground_times, ground_values)
            np.testing.assert_allclose(pitch._values, expected)
            self.assertIsNotNone(pitch._point_at(0.225))
            self.assertIsNone(pitch._point_at(duration + 1.0))
            self.assertFalse(pitch._lod_detailed)
            self.assertFalse(pitch._lod_symbols_visible)
            self.assertFalse(timing._lod_detailed)
            self.assertFalse(recordings._lod_detailed)
            self.assertFalse(intonation._lod_detailed)
            for track, display_count in (
                    (timing, timing._display_bar_count),
                    (recordings, recordings._display_row_count),
                    (intonation, intonation._display_block_count)):
                width = max(32.0, track.getViewBox()
                            .sceneBoundingRect().width())
                self.assertLessEqual(
                    display_count,
                    int(np.ceil(width / fg.PARAMETER_OVERVIEW_BUCKET_PX)) + 2)

            for track in tracks:
                track.setXRange(0.0, 0.4, padding=0)
            pitch._refresh_lod()
            timing._redraw_bars()
            recordings._redraw()
            intonation._redraw()
            self.app.processEvents()

            self.assertTrue(pitch._lod_detailed)
            self.assertTrue(pitch._lod_symbols_visible)
            self.assertTrue(timing._lod_detailed)
            self.assertTrue(recordings._lod_detailed)
            self.assertTrue(intonation._lod_detailed)
            self.assertTrue(recordings._labels)
            self.assertTrue(intonation._labels)
        finally:
            for track in tracks:
                track.close()
                track.deleteLater()
            self.app.processEvents()

    def test_shortcuts_generate_from_text_focus_and_timing_is_undoable(self):
        calls = []
        self.window.on_generate = lambda: calls.append("generate")
        self.window.text.setFocus()
        event = fg.QtGui.QKeyEvent(
            fg.QtCore.QEvent.KeyPress, fg.Qt.Key_R,
            fg.Qt.ControlModifier)

        self.assertTrue(self.window.eventFilter(self.window.text, event))
        self.assertEqual(calls, ["generate"])

        segments = [fg.fc.Segment("pau", 0.0, .1),
                    fg.fc.Segment("a", .1, .2),
                    fg.fc.Segment("pau", .2, .3)]
        syn = fg.fc.Synthesis(
            np.zeros(300, np.float32), 1000, segments,
            text="a", voicebank="test")
        self.window._show_synthesis(syn)
        self.window._on_timing_commit({1: 2.0})
        self.assertAlmostEqual(self.window.waveform.factors()[1], 2.0,
                               places=2)
        self.window.undo_stack.undo()
        self.assertAlmostEqual(self.window.waveform.factors()[1], 1.0,
                               places=2)
        self.window.undo_stack.redo()
        self.assertAlmostEqual(self.window.waveform.factors()[1], 2.0,
                               places=2)

        self.window.shortcuts["play"] = "Alt+P"
        self.window._rebuild_shortcut_lookup()
        self.window._update_shortcut_hints("waveform")
        self.assertIn("Play Alt+P", self.window.shortcut_hint.text())
        self.assertEqual(self.window.shortcuts["duplicate"], "Ctrl+D")

    def test_waveform_shortcuts_preserve_single_phone_region_data(self):
        segments = [fc.Segment("pau", 0, .1),
                    fc.Segment("a", .1, .2),
                    fc.Segment("b", .2, .3),
                    fc.Segment("pau", .3, .4)]
        waveform = self.window.waveform
        waveform.set_synthesis(fc.Synthesis(
            np.arange(400, dtype=np.float32) / 400.0,
            1000, segments, text="a b"))
        waveform._set_selected_range(1, 1)
        original_ids = [segment.uid for segment in waveform.segments]

        self.assertTrue(self.window._shortcut_duplicate())
        self.assertEqual([segment.phone for segment in waveform.segments],
                         ["pau", "a", "a", "b", "pau"])
        self.assertEqual(waveform.selected_range, (2, 2))
        self.assertEqual(len(waveform.base_audio), 5)
        duplicated_ids = [segment.uid for segment in waveform.segments]
        self.assertEqual(len(set(duplicated_ids)), 5)
        self.assertNotEqual(duplicated_ids[1], duplicated_ids[2])
        self.window.undo_stack.undo()
        self.assertEqual([segment.phone for segment in waveform.segments],
                         ["pau", "a", "b", "pau"])
        self.assertEqual([segment.uid for segment in waveform.segments],
                         original_ids)
        self.window.undo_stack.redo()
        self.assertEqual([segment.uid for segment in waveform.segments],
                         duplicated_ids)
        self.window.undo_stack.undo()

        waveform._set_selected_range(2, 2)
        self.assertTrue(self.window._shortcut_copy())
        self.assertTrue(self.window._shortcut_delete(confirm=False))
        self.assertEqual([segment.phone for segment in waveform.segments],
                         ["pau", "a", "pau"])
        self.assertTrue(self.window._shortcut_paste())
        self.assertEqual([segment.phone for segment in waveform.segments],
                         ["pau", "a", "b", "pau"])

    def test_sentence_and_phrase_shortcuts_duplicate_after_selection(self):
        self.window.sentences = [
            self.window._new_sentence_state("one"),
            self.window._new_sentence_state("two")]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)
        self.window._refresh_sentences_view()
        self.window.mode_tabs.setCurrentIndex(1)
        self.window.sentences_view.set_selected_indices([0])

        self.assertTrue(self.window._shortcut_duplicate())
        self.assertEqual([state["text"] for state in self.window.sentences],
                         ["one", "one", "two"])

        state = self.window.sentences[0]
        state["phrases"] = [
            self.window._new_phrase_state("first"),
            self.window._new_phrase_state("second")]
        state["text"] = "first [pau] [pau] second"
        self.window._refresh_sentences_view()
        self.window.sentences_view.set_selected_phrase_keys([(0, 0)])
        self.assertTrue(self.window._shortcut_duplicate())
        self.assertEqual(
            [phrase["text"] for phrase in self.window.sentences[0]["phrases"]],
            ["first", "first", "second"])

    def test_phrase_paste_with_no_sentence_is_a_safe_noop(self):
        phrase = self.window._new_phrase_state("copied phrase")
        self.window._project_clipboard = {
            "kind": "phrases",
            "items": [{"phrase": phrase, "preview": None}],
        }
        self.window.sentences = []
        self.window._active_sentence_index = -1
        self.window._refresh_sentence_selector(-1)
        self.window._refresh_sentences_view()
        self.window.mode_tabs.setCurrentIndex(1)

        self.assertFalse(self.window._shortcut_paste())

    def test_blank_sentences_canvas_starts_rectangle_selection(self):
        canvas = fg.SentenceSelectionCanvas()
        canvas.resize(320, 180)
        canvas.show()
        self.app.processEvents()
        started, moved, finished = [], [], []
        canvas.selectionDragStarted.connect(
            lambda point, modifiers: started.append((point, modifiers)))
        canvas.selectionDragMoved.connect(moved.append)
        canvas.selectionDragFinished.connect(finished.append)
        start = fg.QtCore.QPoint(20, 20)
        end = fg.QtCore.QPoint(90, 70)

        canvas.mousePressEvent(fg.QtGui.QMouseEvent(
            fg.QtCore.QEvent.MouseButtonPress, fg.QtCore.QPointF(start),
            fg.QtCore.QPointF(canvas.mapToGlobal(start)), fg.Qt.LeftButton,
            fg.Qt.LeftButton, fg.Qt.NoModifier))
        canvas.mouseMoveEvent(fg.QtGui.QMouseEvent(
            fg.QtCore.QEvent.MouseMove, fg.QtCore.QPointF(end),
            fg.QtCore.QPointF(canvas.mapToGlobal(end)), fg.Qt.NoButton,
            fg.Qt.LeftButton, fg.Qt.NoModifier))
        canvas.mouseReleaseEvent(fg.QtGui.QMouseEvent(
            fg.QtCore.QEvent.MouseButtonRelease, fg.QtCore.QPointF(end),
            fg.QtCore.QPointF(canvas.mapToGlobal(end)), fg.Qt.LeftButton,
            fg.Qt.NoButton, fg.Qt.NoModifier))

        self.assertEqual(len(started), 1)
        self.assertEqual(len(moved), 1)
        self.assertEqual(len(finished), 1)
        canvas.deleteLater()

    def test_pending_visuals_keep_generate_on_buttons_only(self):
        syn = fc.Synthesis(
            np.ones(300, np.float32) * .1, 1000,
            [fc.Segment("pau", 0, .1), fc.Segment("a", .1, .2),
             fc.Segment("pau", .2, .3)], text="a", voicebank="voice")
        state = self.window.sentences[0]
        state["synthesis"] = syn
        self.window._clear_state_pending(state)
        self.window._show_synthesis(syn)
        self.window._capture_active_sentence()

        self.window._mark_active_pending("rerender", "test")
        self.assertEqual(self.window.waveform._pending_action, "rerender")
        self.assertTrue(bool(
            self.window.btn_rerender.property("renderPending")))
        self.assertFalse(bool(self.window.btn_gen.property("generatePending")))

        self.window._mark_active_pending("generate", "test")
        self.assertEqual(self.window.waveform._pending_action, "")
        self.assertTrue(bool(self.window.btn_gen.property("generatePending")))
        self.assertFalse(self.window.btn_rerender.isEnabled())
        self.assertNotIn("Generate pending", self.window.wf_group.title())

        self.window._clear_state_pending(state)
        with mock.patch.object(
                self.window.waveform.curve, "setPen",
                wraps=self.window.waveform.curve.setPen) as set_pen:
            self.window._mark_active_pending(
                "rerender", "Phoneme timing changed")
            for _index in range(25):
                self.window._refresh_pending_ui()
            self.assertEqual(set_pen.call_count, 1)
        self.assertEqual(
            self.window.waveform.curve.opts["pen"].color().name().upper(),
            "#667FAF")
        self.assertEqual(
            self.window.waveform.boundary_overview.opts[
                "pen"].color().name().upper(), "#D00000")

    def test_sentence_text_revert_clears_neutral_generate_state(self):
        syn = fc.Synthesis(
            np.ones(400, np.float32) * .1, 1000,
            [fc.Segment("pau", 0, .1), fc.Segment("a", .1, .3),
             fc.Segment("pau", .3, .4)],
            text="a statement.", voicebank="voice", phones=["a"])
        self.window.text.setText("a statement.")
        state = self.window.sentences[0]
        state["text"] = "a statement."
        self.window._show_synthesis(syn)
        self.window._commit_rendered_state(syn)
        self.window._capture_active_sentence()
        self.window._refresh_sentences_view()
        row = self.window.sentences_view.row_widgets[0]
        editor = row.findChild(fg.SentenceTextEdit)
        phrase_ids = [phrase["id"] for phrase in state["phrases"]]

        editor.setPlainText("a statement?")
        self.app.processEvents()

        self.assertEqual(self.window._pending_action(state), "generate")
        self.assertEqual(row.property("pending"), "generate")
        self.assertTrue(bool(row.generate_button.property("generatePending")))
        self.assertTrue(row.pending_badge.isHidden())
        self.assertEqual(row.pending_badge.text(), "")
        self.assertFalse(row.play_button.isEnabled())
        self.assertIn(
            'QFrame#sentenceRow[pending="generate"] { background: #E7E6E1; }',
            fg.XP_QSS)
        self.assertNotIn(
            'QLabel#pendingBadge[pending="generate"]', fg.XP_QSS)
        self.assertNotIn("#F4D6CC", fg.XP_QSS)

        editor.setPlainText("a statement.")
        self.app.processEvents()

        self.assertEqual(self.window._pending_action(state), "")
        self.assertEqual(row.property("pending"), "")
        self.assertFalse(bool(row.generate_button.property("generatePending")))
        self.assertTrue(row.pending_badge.isHidden())
        self.assertTrue(row.play_button.isEnabled())
        self.assertTrue(state["rendered"])
        self.assertEqual([phrase["id"] for phrase in state["phrases"]],
                         phrase_ids)

    def test_repeated_pending_edits_advance_render_revision(self):
        state = self.window.sentences[0]

        self.window._set_state_pending(state, "generate", "Text changed")
        first = state["_edit_revision"]
        self.window._set_state_pending(state, "generate", "Text changed")

        self.assertGreater(state["_edit_revision"], first)
        self.window._clear_state_pending(state)
        self.assertEqual(state["_edit_revision"], first + 1)

    def test_sentence_editor_enter_submits_and_shift_enter_adds_phrase_line(self):
        row = fg.SentenceRow(
            0, self.window._new_sentence_state("first phrase"))
        editor = row.findChild(fg.SentenceTextEdit)
        submitted = []
        row.generateRequested.connect(submitted.append)
        row.show()
        self.app.processEvents()
        initial_height = editor.height()
        editor.moveCursor(fg.QtGui.QTextCursor.End)

        fg.QtWidgets.QApplication.sendEvent(
            editor, fg.QtGui.QKeyEvent(
                fg.QtCore.QEvent.KeyPress, fg.Qt.Key_Return,
                fg.Qt.NoModifier))

        self.assertEqual(submitted, [0])
        self.assertEqual(editor.toPlainText(), "first phrase")

        fg.QtWidgets.QApplication.sendEvent(
            editor, fg.QtGui.QKeyEvent(
                fg.QtCore.QEvent.KeyPress, fg.Qt.Key_Return,
                fg.Qt.ShiftModifier))
        editor.insertPlainText("second phrase")
        fg.QtWidgets.QApplication.sendEvent(
            editor, fg.QtGui.QKeyEvent(
                fg.QtCore.QEvent.KeyPress, fg.Qt.Key_Return,
                fg.Qt.ShiftModifier))
        editor.insertPlainText("third phrase")
        self.app.processEvents()

        self.assertEqual(submitted, [0])
        self.assertEqual(
            editor.toPlainText(),
            "first phrase\nsecond phrase\nthird phrase")
        self.assertGreater(editor.minimumHeight(), initial_height)
        self.assertLessEqual(editor.maximumHeight(), editor.MAXIMUM_HEIGHT)
        row.deleteLater()

    def test_broken_pitch_menu_only_pins_exact_active_faults(self):
        waveform = self.window.waveform
        waveform.set_synthesis(fc.Synthesis(
            np.zeros(300, np.float32), 1000,
            [fc.Segment("pau", 0, .1), fc.Segment("a", .1, .2),
             fc.Segment("pau", .2, .3)],
            fault_events=[{"kind": "pitch_glitch", "segment": 1,
                           "phone": "a", "broken_hz": 91.25}]))
        waveform._set_selected_range(1, 1)
        captured = []

        def capture(menu, *_args):
            captured.append([action.text() for action in menu.actions()])

        with mock.patch.object(fg.QtWidgets.QMenu, "exec_", new=capture):
            waveform.set_fault_mode_active(False)
            waveform._selection_menu(.15, fg.QtCore.QPointF())
            waveform.set_fault_mode_active(True)
            waveform._selection_menu(.15, fg.QtCore.QPointF())

        self.assertFalse(any("pitch" in text.lower()
                             for text in captured[0]))
        self.assertTrue(any("Pin selected broken pitch fault" in text
                            for text in captured[1]))
        requested = []
        waveform.faultTargetRequested.connect(requested.append)
        waveform._fault_mode_active = True
        self.window._set_pitch_fault_target([
            {"kind": "pitch_glitch", "segment": 1,
             "phone": "a", "broken_hz": 91.25}])
        self.assertEqual(
            self.window._fault_mode()["pitch_glitch_pins"][0]["broken_hz"],
            91.25)

    def test_gain_controls_are_resettable_peak_safe_and_show_pending_state(self):
        self.assertFalse(self.window.speech_gain.isEnabled())
        self.assertFalse(self.window.sentences_view.gain.isEnabled())
        samples = np.ones(400, np.float32) * .5
        segments = [fc.Segment("pau", 0, .1),
                    fc.Segment("a", .1, .3),
                    fc.Segment("pau", .3, .4)]
        self.window.speech_gain.setEnabled(True)
        self.window.speech_gain.set_value(12.0)
        syn = self.window._apply_output_faults(fc.Synthesis(
            samples.copy(), 1000, segments, text="a", voicebank="voice"))
        self.window._show_synthesis(syn)
        self.window._capture_active_sentence()

        self.assertTrue(self.window.speech_gain.isEnabled())
        self.assertTrue(self.window.sentences_view.gain.isEnabled())
        self.assertAlmostEqual(self.window.speech_gain.spin.maximum(),
                               20 * np.log10(2), places=1)
        self.assertLessEqual(float(np.max(np.abs(syn.samples))), 1.0)

        self.window.speech_gain.set_value(3.0, emit=True)
        self.assertTrue(bool(self.window.speech_gain.property("gainPending")))
        self.window._on_allow_clipping_changed(True)
        self.assertEqual(self.window.speech_gain.spin.maximum(), 12.0)
        self.assertEqual(self.window.sentences_view.gain.spin.maximum(), 12.0)

        for index in range(20):
            value = -2.0 + index * 0.1
            self.window.speech_gain.set_value(value, emit=True)
            self.window.sentences_view.gain.set_value(value, emit=True)

        emitted = []
        control = fg.GainControl()
        control.valueChanged.connect(emitted.append)
        control.set_allow_clipping(True, emit=False)
        control.set_value(12.0, emit=False)
        control.set_audio_state(True, peak=.5, applied_gain_db=0.0)
        control.set_allow_clipping(False, emit=False)
        self.assertEqual(emitted, [])
        self.assertAlmostEqual(control.value(), 20 * np.log10(2), places=1)
        control.deleteLater()

        self.window.output_gain_slider.setValue(-120)
        event = fg.QtGui.QMouseEvent(
            fg.QtCore.QEvent.MouseButtonDblClick, fg.QtCore.QPointF(4, 4),
            fg.Qt.LeftButton, fg.Qt.LeftButton, fg.Qt.NoModifier)
        self.window.output_gain_slider.mouseDoubleClickEvent(event)
        self.assertEqual(self.window.output_gain_slider.value(), 0)

    def test_disabled_menu_items_have_a_distinct_neutral_background(self):
        menu = fg.QtWidgets.QMenu()
        try:
            menu.setStyleSheet(fg.XP_QSS)
            enabled = menu.addAction("Available command")
            disabled = menu.addAction("Unavailable command")
            disabled.setEnabled(False)
            menu.ensurePolished()
            menu.resize(menu.sizeHint())
            menu.show()
            self.app.processEvents()

            image = menu.grab().toImage()
            sample_x = menu.width() - 8
            enabled_color = image.pixelColor(
                sample_x, menu.actionGeometry(enabled).center().y()).name()
            disabled_color = image.pixelColor(
                sample_x, menu.actionGeometry(disabled).center().y()).name()
            self.assertEqual(enabled_color, "#ffffff")
            self.assertEqual(disabled_color, "#e2e0da")
            self.assertNotEqual(enabled_color, disabled_color)
            enabled_rect = menu.actionGeometry(enabled)
            disabled_rect = menu.actionGeometry(disabled)
            self.assertEqual(enabled_rect.left(), disabled_rect.left())
            self.assertEqual(enabled_rect.width(), disabled_rect.width())
            self.assertIn("QMenu::item { min-height: 18px; padding:",
                          fg.XP_QSS)
            self.assertNotIn("font-style: italic", fg.XP_QSS)
        finally:
            menu.close()
            menu.deleteLater()

    def test_fast_parameter_sweeps_interpolate_and_shift_latches_value(self):
        spans = [(index * .1, (index + 1) * .1) for index in range(8)]
        phones = ["a"] * len(spans)
        pitch = fg.PitchTrack()
        pitch.set_data(spans, phones,
                       [(0.0, 140.0), (.8, 220.0)], base=180.0)
        first = fg.QtCore.QPointF(.02, 130.0)
        pitch._edit(first, previous=first)
        pitch._edit(fg.QtCore.QPointF(.78, 230.0), previous=first)
        values = [value for _time, value in pitch.targets()]
        self.assertGreater(len([value for value in values
                                if 135.0 < value < 225.0]), 3)

        pitch._latched_value = 175.0
        pitch._edit(fg.QtCore.QPointF(.78, 400.0),
                    previous=fg.QtCore.QPointF(.02, 20.0))
        self.assertTrue(all(abs(value - 175.0) < 1e-6
                            for _time, value in pitch.targets()))

        timing = fg.TimingTrack()
        timing.set_segments(spans, [1.0] * len(spans), phones)
        timing._latched_value = .5
        timing._paint(fg.QtCore.QPointF(.78, -1.5), previous_x=.02)
        self.assertTrue(all(abs(value - 2 ** .5) < 1e-6
                            for value in timing.factors()))

        timing.set_segments(
            spans, [2.0, 0.5, 1.5, 0.75] * 2, phones
        )
        timing._paint(
            fg.QtCore.QPointF(.78, 1.5), previous_x=.02, reset=True
        )
        self.assertTrue(all(abs(value - 1.0) < 1e-6
                            for value in timing.factors()))
        self.assertEqual(
            timing._touched, {index: 1.0 for index in range(len(spans))}
        )
        pitch.deleteLater()
        timing.deleteLater()

    def test_moraic_nasal_uses_vowel_timing_filter_without_global_reclass(self):
        timing = fg.TimingTrack()
        try:
            spans = [(0.0, 0.12), (0.12, 0.20)]
            timing.set_segments(
                spans, [1.0, 1.0], ["nn", "n"],
                ["moraic_nasal", "consonant"],
            )
            timing.set_filter(consonants=True, vowels=False)
            self.assertFalse(timing._editable(0))
            self.assertTrue(timing._editable(1))

            timing.set_filter(consonants=False, vowels=True)
            self.assertTrue(timing._editable(0))
            self.assertFalse(timing._editable(1))

            timing.set_segments(spans, [1.0, 1.0], ["nn", "n"])
            timing.set_filter(consonants=True, vowels=False)
            self.assertTrue(timing._editable(0))
        finally:
            timing.deleteLater()

    def test_timing_right_drag_resets_crossed_regions_once(self):
        timing = fg.TimingTrack()
        try:
            spans = [(index * .1, (index + 1) * .1)
                     for index in range(4)]
            timing.set_segments(
                spans, [2.0, 0.5, 1.5, 0.75], ["a"] * len(spans)
            )
            commits = []
            timing.factorsCommitted.connect(
                lambda changes: commits.append(dict(changes))
            )

            press = mock.Mock()
            press.button.return_value = fg.Qt.RightButton
            with mock.patch.object(
                    timing, "_view_pos",
                    return_value=fg.QtCore.QPointF(.02, 0.0)):
                timing.mousePressEvent(press)
            self.assertEqual(timing._drag_mode, "reset")

            move = mock.Mock()
            with mock.patch.object(
                    timing, "_view_pos",
                    return_value=fg.QtCore.QPointF(.38, 0.0)):
                timing.mouseMoveEvent(move)
            release = mock.Mock()
            timing.mouseReleaseEvent(release)

            self.assertTrue(all(abs(value - 1.0) < 1e-6
                                for value in timing.factors()))
            self.assertEqual(commits, [{index: 1.0 for index in range(4)}])
            self.assertEqual(timing._drag_mode, "")
        finally:
            timing.deleteLater()

    def test_voicing_track_exposes_each_analysis_frame_without_handles(self):
        track = fg.VoicingTrack()
        try:
            spans = [(0.0, 0.08), (0.08, 0.16)]
            phones = ["u", "a"]
            ground = [
                (round(index * 0.008, 6), 0.45 + index * 0.01)
                for index in range(21)
            ]
            track.set_data(spans, phones, ground, [], 1.0)

            times = [time for time, _value in track.targets()]
            self.assertEqual(times, [time for time, _value in ground])
            self.assertAlmostEqual(times[1] - times[0], 0.008)
            track.resize(1000, 180)
            track.show()
            self.app.processEvents()
            track._refresh_lod()
            self.assertFalse(track._lod_symbols_visible)
            self.assertLess(len(track._visual_ground_x), len(track._ground_x))
        finally:
            track.close()
            track.deleteLater()

    def test_vocal_tract_track_has_at_most_two_controls_per_phone(self):
        track = fg.VocalTractTrack()
        try:
            spans = [(0.0, 0.12), (0.12, 0.28), (0.28, 0.60)]
            track.set_data(spans, ["k", "e", "pau"], [], [])
            times = [time for time, _value in track.targets()]
            self.assertEqual(times, [0.0, 0.12, 0.28, 0.60])
            for start, end in spans:
                controls = [time for time in times
                            if start - 1e-9 <= time <= end + 1e-9]
                self.assertLessEqual(len(controls), 2)
        finally:
            track.close()
            track.deleteLater()

    def test_voicing_refresh_keeps_generated_and_manual_curves_distinct(self):
        segments = [fc.Segment("pau", 0.0, 0.04),
                    fc.Segment("u", 0.04, 0.20),
                    fc.Segment("pau", 0.20, 0.24)]
        source = [(0.0, 0.0), (0.04, 0.15), (0.08, 0.92),
                  (0.16, 0.74), (0.20, 0.12), (0.24, 0.0)]
        generated = [(0.0, 0.0), (0.04, 0.15), (0.08, 0.82),
                     (0.12, 0.16), (0.18, 0.70), (0.24, 0.0)]
        override = [(0.0, 0.0), (0.08, 0.66), (0.12, 0.42),
                    (0.20, 0.1), (0.24, 0.0)]
        syn = fc.Synthesis(
            np.zeros(3840, np.float32), 16000, segments,
            source_voicing_targets=source,
            generated_voicing_targets=generated,
            voicing_override=override,
            voicing_mode="curve",
        )

        self.window._show_synthesis(syn)
        self.assertEqual(self.window.voicing_track._ground, generated)
        self.assertNotEqual(
            self.window.voicing_track._ground,
            self.window.voicing_track.targets(),
        )

        self.window._sync_timing_track(reset=False)
        self.window._sync_timing_track(reset=True)

        self.assertEqual(self.window.current.generated_voicing_targets,
                         generated)
        self.assertEqual(self.window.voicing_track._ground, generated)
        self.assertEqual(self.window.current.voicing_mode, "curve")
        self.assertTrue(self.window.current.voicing_override)
        self.assertNotEqual(self.window.current.voicing_override, generated)

    def test_manual_pitch_curve_remaps_with_stretched_phone_geometry(self):
        pitch = fg.PitchTrack()
        try:
            pitch.set_data(
                [(0.0, 1.0), (1.0, 2.0)],
                ["a", "i"],
                [(0.0, 150.0), (2.0, 150.0)],
                [(0.0, 150.0), (0.5, 210.0),
                 (1.5, 190.0), (2.0, 150.0)],
            )

            pitch.update_geometry(
                [(0.0, 2.0), (2.0, 3.0)],
                ["a", "i"],
                [(0.0, 150.0), (3.0, 150.0)],
            )

            targets = dict(pitch.targets())
            self.assertEqual(list(targets), [0.0, 1.0, 2.5, 3.0])
            self.assertAlmostEqual(targets[1.0], 210.0)
            self.assertAlmostEqual(targets[2.5], 190.0)
        finally:
            self.app.processEvents()
            pitch.close()
            pitch.deleteLater()

    def test_pitch_delete_keeps_surviving_deviations_by_segment_identity(self):
        pitch = fg.PitchTrack()
        try:
            old_spans = [
                (0.0, 0.10), (0.10, 0.30), (0.30, 0.50),
                (0.50, 0.60), (0.60, 0.80), (0.80, 0.90),
            ]
            phones = ["pau", "r", "ae", "b", "ih", "pau"]
            ids = ["lead", "rabbit-r", "rabbit-ae", "deleted-b",
                   "rabbit-ih", "tail"]
            control_times = pitch._control_times(old_spans, phones)
            generated = [
                (time, 165.0 + time * 20.0) for time in control_times
            ]
            multipliers = {
                (old_spans[1][0] + old_spans[1][1]) * .5: 1.12,
                (old_spans[4][0] + old_spans[4][1]) * .5: 0.86,
            }
            override = [
                (time, value * multipliers.get(time, 1.0))
                for (time, value) in generated
            ]
            pitch.set_data(
                old_spans, phones, generated, override,
                segment_ids=ids)

            new_spans = [
                (0.0, 0.10), (0.10, 0.30), (0.30, 0.50),
                (0.50, 0.70), (0.70, 0.80),
            ]
            new_phones = ["pau", "r", "ae", "ih", "pau"]
            new_ids = ["lead", "rabbit-r", "rabbit-ae", "rabbit-ih", "tail"]
            old_segments = [
                fc.Segment(phone, start, end, uid=uid)
                for (start, end), phone, uid in
                zip(old_spans, phones, ids)
            ]
            new_segments = [
                fc.Segment(phone, start, end, uid=uid)
                for (start, end), phone, uid in
                zip(new_spans, new_phones, new_ids)
            ]
            new_ground = fc.remap_targets_aligned(
                generated, old_segments, new_segments)

            pitch.update_geometry(
                new_spans, new_phones, new_ground,
                segment_ids=new_ids)

            controls = dict(pitch.targets())
            r_mid = .20
            ih_mid = .60
            sample = pitch._sample_many
            r_ground = sample(new_ground, [r_mid], 165.0)[0]
            ih_ground = sample(new_ground, [ih_mid], 165.0)[0]
            self.assertAlmostEqual(controls[r_mid] / r_ground, 1.12)
            self.assertAlmostEqual(controls[ih_mid] / ih_ground, 0.86)

            stable = list(pitch.targets())
            for _cycle in range(8):
                pitch.update_geometry(
                    new_spans, new_phones, new_ground,
                    segment_ids=new_ids)
            np.testing.assert_allclose(
                pitch.targets(), stable, rtol=0.0, atol=1.0e-10)
        finally:
            self.app.processEvents()
            pitch.close()
            pitch.deleteLater()

    def test_sentence_phone_delete_and_timing_edit_do_not_accumulate_pitch(self):
        segments = [
            fc.Segment("pau", 0.00, 0.08, uid="lead"),
            fc.Segment("dh", 0.08, 0.16, uid="the-dh"),
            fc.Segment("ax", 0.16, 0.28, uid="the-ax"),
            fc.Segment("r", 0.28, 0.36, uid="rabbit-r"),
            fc.Segment("ae", 0.36, 0.54, uid="rabbit-ae"),
            fc.Segment("b", 0.54, 0.62, uid="deleted-b"),
            fc.Segment("ih", 0.62, 0.76, uid="rabbit-ih"),
            fc.Segment("t", 0.76, 0.84, uid="rabbit-t"),
            fc.Segment("pau", 0.84, 0.92, uid="tail"),
        ]
        entries = [(segment.phone, segment.dur) for segment in segments]
        generated = fc.anchor_phrase_targets(
            entries,
            [(0.08, 181.0), (0.20, 188.0), (0.32, 176.0),
             (0.45, 194.0), (0.58, 183.0), (0.69, 171.0),
             (0.80, 166.0)],
            165.0,
        )
        synthesis = fc.Synthesis(
            np.zeros(920, np.float32), 1000, segments,
            text="the rabbit", lang="en", voicebank="fixture",
            targets=generated, generated_targets=generated,
        )
        self.window._show_synthesis(synthesis)
        self.window.sentences[0]["rendered"] = True

        edited_uid = "rabbit-ae"
        edited_segment = next(
            segment for segment in self.window.waveform.segments
            if segment.uid == edited_uid)
        edited_time = (edited_segment.start + edited_segment.end) * .5
        self.window.pitch_track._edit(
            fg.QtCore.QPointF(edited_time, 225.0))
        self.window._on_pitch_commit(self.window.pitch_track.targets())

        def deviations():
            result = {}
            track = self.window.pitch_track
            for segment in self.window.waveform.segments:
                midpoint = (segment.start + segment.end) * .5
                position = track._point_at(midpoint)
                if position is None:
                    continue
                ground = track._sample_ground(track._times[position], 165.0)
                result[segment.uid] = (
                    track._values[position] / max(1.0e-9, ground)
                )
            return result

        before = deviations()
        delete_index = next(
            index for index, segment in
            enumerate(self.window.waveform.segments)
            if segment.uid == "deleted-b")
        self.window.waveform._delete_phone(delete_index)
        after_delete = deviations()
        for uid in set(before) & set(after_delete):
            self.assertAlmostEqual(after_delete[uid], before[uid], places=9)

        stretch_index = next(
            index for index, segment in
            enumerate(self.window.waveform.segments)
            if segment.uid == "rabbit-ih")
        for factor in (1.25, 0.85, 1.40, 1.0):
            self.window.waveform.set_factor(stretch_index, factor)
        after_timing = deviations()
        for uid in set(after_delete) & set(after_timing):
            self.assertAlmostEqual(
                after_timing[uid], after_delete[uid], places=9)

    def test_local_pitch_edit_preserves_dense_generated_contour(self):
        segments = [
            fc.Segment("a", 0.0, 1.0),
            fc.Segment("i", 1.0, 2.0),
            fc.Segment("u", 2.0, 3.0),
            fc.Segment("e", 3.0, 4.0),
        ]
        generated = [
            (index * 0.25, 145.0 + index * 2.0 +
             (5.0 if index % 3 == 0 else 0.0))
            for index in range(17)
        ]
        synthesis = fc.Synthesis(
            np.zeros(4000, np.float32), 1000, segments,
            text="aiue", voicebank="test",
            targets=generated, generated_targets=generated,
        )
        self.window._show_synthesis(synthesis)
        pitch = self.window.pitch_track
        point = fg.QtCore.QPointF(2.5, 230.0)
        pitch._edit(point, previous=point)

        sparse = pitch.targets()
        rendered = pitch.render_targets()
        self.assertGreater(len(rendered), len(sparse))
        rendered_by_time = dict(rendered)
        for time, value in generated:
            self.assertIn(time, rendered_by_time)
            if time <= 1.5 or time >= 3.5:
                self.assertAlmostEqual(rendered_by_time[time], value)
        self.assertAlmostEqual(rendered_by_time[2.5], 230.0)

        self.window._on_pitch_commit(sparse)
        stored = dict(self.window.current.pitch_override)
        for time, value in generated:
            self.assertIn(time, stored)
            if time <= 1.5 or time >= 3.5:
                self.assertAlmostEqual(stored[time], value)

    def test_unedited_pitch_source_prefers_actual_rendered_contour(self):
        stale = [(0.1, 205.0), (0.3, 195.0)]
        rendered = [(0.1, 170.0), (0.3, 160.0)]
        synthesis = fc.Synthesis(
            np.zeros(400, np.float32),
            1000,
            [
                fc.Segment("pau", 0.0, 0.1),
                fc.Segment("a", 0.1, 0.3),
                fc.Segment("pau", 0.3, 0.4),
            ],
            targets=rendered,
            generated_targets=stale,
        )

        self.assertEqual(
            self.window._synthesis_pitch_source(synthesis),
            rendered,
        )
        synthesis._pitch_reset_pending = True
        self.assertEqual(
            self.window._synthesis_pitch_source(synthesis),
            stale,
        )
        synthesis._pitch_reset_pending = False
        synthesis.pitch_mode = "curve"
        synthesis.pitch_override = list(rendered)
        self.assertEqual(
            self.window._synthesis_pitch_source(synthesis),
            stale,
        )

    def test_pitch_middle_pan_uses_stable_pixel_delta(self):
        pitch = fg.PitchTrack()
        pitch.resize(800, 200)
        pitch.show()
        pitch.set_data([(0.0, 1.0)], ["a"],
                       [(0.0, 160.0), (1.0, 180.0)])
        self.app.processEvents()
        pitch._pan_start_pixel_y = 100.0
        pitch._pan_start_center = 170.0
        pitch._pan_start_span = 240.0
        pitch._pan_height = 200.0

        pitch._pan_to_pixel_y(120.0)
        first = pitch._view_center
        pitch._pan_to_pixel_y(120.0)

        self.assertAlmostEqual(first, 194.0)
        self.assertAlmostEqual(pitch._view_center, first)
        pitch.deleteLater()

    def test_shift_wheel_scroll_path_does_not_move_selection(self):
        viewbox = fg.SelectableWaveformViewBox()
        viewbox.setXRange(0.0, 10.0, padding=0)
        selection_moves = []
        viewbox.selectionMoveDragged.connect(selection_moves.append)

        viewbox._scroll_x(-120.0)

        left, right = viewbox.viewRange()[0]
        self.assertAlmostEqual(right - left, 10.0)
        self.assertGreater(left, 0.0)
        self.assertEqual(selection_moves, [])

    def test_pitch_navigation_is_vertical_and_controls_pitch_scale(self):
        self.window.parameter_mode.setCurrentIndex(
            self.window.parameter_mode.findData("pitch"))
        synthesis = fc.Synthesis(
            np.zeros(6000, np.float32), 1000,
            [fc.Segment("aa", 0.0, 6.0)])
        self.window.waveform.set_synthesis(synthesis)
        self.window.waveform.plot.setXRange(1.5, 3.5, padding=0)
        self.window.show()
        self.app.processEvents()
        self.window._sync_pitch_navigator()
        self.assertEqual(self.window.pitch_scroll.orientation(), fg.Qt.Vertical)
        self.assertGreater(self.window.pitch_scroll.height(),
                           self.window.pitch_scroll.width())
        initial_zoom = self.window.pitch_track.zoom_level()
        self.window.pitch_zoom_in.click()
        self.assertEqual(self.window.pitch_track.zoom_level(), initial_zoom + 1)
        center = min(int(fc.PITCH_MAX_HZ),
                     self.window.pitch_scroll.value() + 10)
        self.window.pitch_scroll.setValue(center)
        self.assertAlmostEqual(self.window.pitch_track.view_center(), center)
        self.assertEqual(self.window.waveform.plot.viewRange()[0], [1.5, 3.5])

    def test_waveform_auto_adjust_is_a_one_shot_full_duration_fit(self):
        synthesis = fc.Synthesis(
            np.zeros(4000, np.float32), 1000,
            [fc.Segment("aa", 0.0, 4.0)])
        waveform = self.window.waveform
        waveform.set_synthesis(synthesis)
        waveform.plot.setXRange(1.0, 1.5, padding=0)

        viewbox = waveform.plot.getViewBox()
        viewbox.enableAutoRange()
        self.app.processEvents()

        left, right = viewbox.viewRange()[0]
        self.assertLessEqual(left, 0.0)
        self.assertGreaterEqual(right, 4.0)
        self.assertEqual(viewbox.autoRangeEnabled(), [False, False])
        self.assertEqual(viewbox.viewRange()[1], [-1.05, 1.05])

    def test_pitch_pause_nodes_and_lines_use_neutral_layers(self):
        pitch = fg.PitchTrack()
        try:
            pitch.resize(700, 160)
            pitch.show()
            pitch.setXRange(0.0, 0.7, padding=0)
            pitch.set_data(
                [(0.0, 0.2), (0.2, 0.5), (0.5, 0.7)],
                ["aa", "pau", "aa"],
                [(0.0, 150.0), (0.7, 180.0)],
            )
            self.app.processEvents()

            pause_x, _pause_y = pitch._pause_points.getData()
            speech_x, _speech_y = pitch._override_points.getData()
            self.assertTrue(len(pause_x))
            self.assertTrue(all(0.2 <= value <= 0.5 for value in pause_x))
            self.assertTrue(any(value < 0.2 or value > 0.5
                                for value in speech_x))
            self.assertTrue(np.isfinite(pitch._pause_curve.xData).any())
            self.assertEqual(
                pitch._pause_curve.opts["pen"].color().hsvSaturation(), 0)
            self.assertEqual(
                pitch._pause_points.opts["brush"].color().hsvSaturation(), 0)
        finally:
            pitch.close()
            pitch.deleteLater()

    def test_zoomed_out_pitch_curve_keeps_final_sentence_line(self):
        pitch = fg.PitchTrack()
        try:
            count = 420
            step = 0.01
            spans = [(index * step, (index + 1) * step)
                     for index in range(count)]
            phones = ["aa"] * (count - 2) + ["pau", "pau"]
            generated = [
                (index * step, 165.0 + 8.0 * math.sin(index * 0.03))
                for index in range(count + 1)
            ]
            pitch.resize(360, 150)
            pitch.show()
            pitch.setXRange(0.0, count * step, padding=0)
            pitch.set_data(spans, phones, generated)
            pitch._refresh_lod()
            self.app.processEvents()

            speech_x = np.asarray(pitch._override_curve.xData, np.float64)
            pause_x = np.asarray(pitch._pause_curve.xData, np.float64)
            final_speech_edge = spans[-2][0]
            self.assertFalse(pitch._lod_detailed)
            self.assertAlmostEqual(
                float(np.nanmax(speech_x)), final_speech_edge, places=6)
            self.assertAlmostEqual(
                float(np.nanmax(pause_x)), spans[-1][1], places=6)
        finally:
            pitch.close()
            pitch.deleteLater()

    def test_recording_track_calls_vowel_to_vowel_edge_vv(self):
        track = fg.RecordingTrack()
        dialog = None
        try:
            segments = [fc.Segment("a", 0.0, 0.1),
                        fc.Segment("i", 0.1, 0.2)]
            track.set_data(segments, {"a-i": [{
                "id": "jc_fixture",
                "left_name": "a",
                "role": "vcv_mora",
                "family": "vcv",
                "transition_kind": "vv",
                "alias": "a い",
            }]}, {0: "a"}, {})

            self.assertIn("VV", track._rows[0]["label"])
            self.assertNotIn("VCV", track._rows[0]["label"])
            dialog = fg.SourcePitchmarkDialog({
                "pair": "a-i",
                "wav_name": "unit.wav",
                "samples": np.zeros(100, np.float32),
                "sr": 1000,
                "pitchmarks": [0.01, 0.02, 0.03],
                "f0_track": [
                    (0.0, 0.0), (0.015, 101.0), (0.025, 100.0),
                ],
                "epoch_f0_track": [(0.015, 100.0), (0.025, 100.0)],
                "f0_track_kind": "analyzed",
                "f0_source": "world-harvest-stonemask",
                "discontinuities": [],
            })
            self.assertIn("a-i", dialog.windowTitle())
            self.assertIn("no large local period jumps", dialog.summary.text())
            self.assertIn("analyzed F0", dialog.summary.text())
            self.assertIn("world-harvest-stonemask", dialog.summary.text())
            self.assertIsNotNone(dialog.waveform_plot)
            self.assertIsNotNone(dialog.f0_plot)
        finally:
            if dialog is not None:
                dialog.close()
                dialog.deleteLater()
            track.close()
            track.deleteLater()

    def test_recording_track_uses_structural_cl_source_pairs(self):
        track = fg.RecordingTrack()
        try:
            segments = [
                fc.Segment("i", 0.0, 0.1),
                fc.Segment("cl", 0.1, 0.18),
                fc.Segment("s", 0.18, 0.28),
                fc.Segment("o", 0.28, 0.4),
            ]
            inventory = {
                "i-s": [{
                    "id": "vc",
                    "left_name": "i",
                    "role": "vc_transition",
                }],
                "s-s": [{
                    "id": "hold",
                    "left_name": "s",
                    "role": "structural_consonant_hold",
                }],
                "s-o": [{
                    "id": "cv",
                    "left_name": "s",
                    "role": "mora_cv",
                }],
            }

            track.set_data(
                segments, inventory, {}, {},
                source_phones=["i", "s", "s", "o"],
            )

            self.assertEqual(track._rows[0]["display_pair"], "i-cl")
            self.assertEqual(track._rows[0]["pair"], "i-s")
            self.assertEqual(track._rows[1]["display_pair"], "cl-s")
            self.assertEqual(track._rows[1]["pair"], "s-s")
            self.assertEqual(track._rows[1]["choice"]["role"],
                             "structural_consonant_hold")
            self.assertIn("s-s source", track._rows[1]["label"])
            self.assertNotIn("cl-s", {
                row["pair"] for row in track._rows
            })
        finally:
            track.close()
            track.deleteLater()

    def test_structural_cl_timing_role_is_not_japanese_only(self):
        segments = [
            fc.Segment("i", 0.0, 0.1),
            fc.Segment("cl", 0.1, 0.18),
            fc.Segment("s", 0.18, 0.28),
        ]
        synthesis = fc.Synthesis(
            np.zeros(280, np.float32), 1000, copy.deepcopy(segments),
            lang="en", phones=["i", "cl", "s"],
            render_phones=["i", "s", "s"],
            special_phone_realizations=[{
                "index": 1,
                "phone": "cl",
                "mode": "anticipatory_consonant",
                "source_phone": "s",
                "status": "resolved",
            }],
        )
        self.window.current = synthesis
        self.window.waveform.set_synthesis(synthesis)

        with mock.patch.object(
                self.window, "_current_lang_code", return_value="en"):
            roles = self.window._japanese_timing_roles()

        self.assertEqual(roles, ["", "structural_vc", ""])

    def test_join_loudness_dialog_aligns_phones_units_collars_and_curve(self):
        rate = 8000
        times = np.arange(int(0.8 * rate), dtype=np.float64) / rate
        samples = 0.05 * np.sin(2.0 * np.pi * 440.0 * times)
        samples[int(0.4 * rate):] *= 6.0
        segments = [
            fc.Segment("a", 0.0, 0.2),
            fc.Segment("b", 0.2, 0.6),
            fc.Segment("c", 0.6, 0.8),
        ]
        alternatives = {
            "a-b": [{
                "left_name": "a_u1", "wav": "incoming.wav",
                "join_conditioning": {
                    "effective_end_collar_ms": 15.0},
            }],
            "b-c": [{
                "left_name": "b_u2", "wav": "outgoing.wav",
                "join_conditioning": {
                    "effective_start_collar_ms": 12.0},
            }],
        }
        diagnostic = fg.diphone_loudness.analyze_rendered_joins(
            samples, rate, segments,
            target_pitchmarks=np.arange(0.0, 0.8, 1.0 / 440.0),
            splice_records=[{
                "unit_index": 0, "segment_index": 1, "time": 0.4,
                "handoff_start": 0.398, "handoff_end": 0.402,
                "position_source": "festival-us-map", "estimated": False,
                "crossover_active": True,
                "crossover_requested_left_ms": 20.0,
                "crossover_requested_right_ms": 20.0,
                "crossover_effective_ms": 34.0,
                "crossover_epoch_intervals": 15,
                "crossover_start": 0.383,
                "crossover_end": 0.417,
                "crossover_context": "vowel",
                "crossover_reason": "context-capped",
            }],
            selected_units={0: "a_u1", 1: "b_u2"},
            alternatives=alternatives)
        diagnostic["frame_trajectory_records"] = [{
            "target_index": 12,
            "time": .42,
            "previous_source_frame": 18,
            "source_frame": 22,
            "centre_offset_samples": -6,
            "original_correlation": .24,
            "corrected_correlation": .95,
            "correlation_improvement": .71,
            "phone": "b",
            "reason": "phase-reference-corrected",
        }]
        dialog = fg.JoinLoudnessDialog(
            diagnostic, samples, focus_edge=0,
            requested_join_settings={
                "mode": "symmetric", "window_factor": 1.08},
            effective_join_settings={
                "window_symmetric": True, "window_factor": 1.08,
                "requested_crossover_ms": 40.0,
                "crossover_ms": 40.0,
                "runtime": "native-crossover",
                "source": "manual-window"},
            editable=True)
        try:
            dialog.show()
            self.app.processEvents()
            self.assertEqual(dialog.table_model.rowCount(), 1)
            self.assertEqual(dialog.table.currentIndex().row(), 0)
            self.assertEqual(len(dialog.segments), 3)
            self.assertEqual(len(dialog.units), 2)
            self.assertIn("1 flagged", dialog.summary.text())
            self.assertIn("1 exact", dialog.summary.text())
            self.assertIn(
                "1 phase-stabilized epochs", dialog.summary.text())
            self.assertIs(fg.JoinLoudnessDialog,
                          fg.JoinDiscontinuityDialog)
            self.assertEqual(dialog.tabs.count(), 3)
            self.assertEqual(dialog.tabs.tabText(2), "Source trajectory")
            self.assertEqual(dialog.trajectory_table.rowCount(), 1)
            self.assertIsNotNone(dialog.trajectory_markers)
            self.assertIn("festival-us-map", dialog.detail_summary.text())
            self.assertIn("phones a b c", dialog.detail_summary.text())
            self.assertIsNotNone(dialog.waveform_plot)
            self.assertIsNotNone(dialog.span_plot)
            self.assertIsNotNone(dialog.loudness_plot)
            self.assertIsNotNone(dialog.local_waveform_plot)
            self.assertIsNotNone(dialog.difference_plot)
            self.assertIsNotNone(dialog.raw_period_plot)
            self.assertIsNotNone(dialog.normalised_period_plot)
            self.assertIsNotNone(dialog.aligned_period_plot)
            self.assertIsNotNone(dialog.rms_plot)
            self.assertIsNotNone(dialog.f0_plot)
            self.assertIsNotNone(dialog.spectral_plot)
            self.assertIsNotNone(dialog.formant_plot)
            self.assertIsNotNone(dialog.spectral_envelope_plot)
            self.assertIsNotNone(dialog.formant_balance_plot)
            self.assertTrue(dialog.apply_window_button.isEnabled())
            self.assertEqual(
                dialog.join_window_mode.currentData(), "symmetric")
            self.assertAlmostEqual(
                dialog.join_window_factor.value(), 1.08)
            self.assertAlmostEqual(
                dialog.join_crossover_ms.value(), 40.0)
            self.assertIn(
                "every pitchmark in the sentence",
                dialog.join_window_explanation.text())
            self.assertIn(
                "Milliseconds stay authoritative",
                dialog.crossover_explanation.text())
            self.assertIsNotNone(dialog._window_left_handle)
            self.assertIsNotNone(dialog._window_right_handle)
            self.assertIsNotNone(dialog._crossover_left_handle)
            self.assertIsNotNone(dialog._crossover_right_handle)
            self.assertIsNotNone(dialog._rendered_crossover_region)
            right_period = diagnostic["joins"][0][
                "right_period_seconds"]
            dialog._window_right_handle.setValue(
                0.4 + right_period * 1.2)
            self.app.processEvents()
            self.assertAlmostEqual(
                dialog.requested_join_settings()["window_factor"],
                1.2, places=2)
            dialog._crossover_right_handle.setValue(0.43)
            self.app.processEvents()
            crossover = dialog.requested_join_settings()
            self.assertAlmostEqual(
                crossover["crossover_overrides"]["0"]["left_ms"],
                20.0, places=1)
            self.assertAlmostEqual(
                crossover["crossover_overrides"]["0"]["right_ms"],
                30.0, places=1)
            left, right = dialog.waveform_plot.viewRange()[0]
            self.assertLess(left, 0.4)
            self.assertGreater(right, 0.4)
            self.assertLess(dialog._wave_cache.last_output_points,
                            len(samples))
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_join_table_defaults_to_render_order_and_can_rank_by_severity(self):
        model = fg.JoinDiagnosticTableModel([
            {"segment_index": 2, "time": .3, "severity_rank": 1},
            {"segment_index": 1, "time": .2, "severity_rank": 2},
        ])
        self.assertEqual(
            [row["segment_index"] for row in model.rows], [1, 2])
        model.set_order("severity")
        self.assertEqual(
            [row["segment_index"] for row in model.rows], [2, 1])

    def test_recording_block_focuses_join_inside_the_same_phone(self):
        joins = []
        for segment_index, when, rank in ((1, .15, 2), (2, .25, 1)):
            joins.append({
                "segment_index": segment_index,
                "time": when,
                "overlap_start": when - .005,
                "overlap_end": when + .005,
                "severity_rank": rank,
                "severity_score": float(3 - rank),
                "dominant_issue": "OK",
                "voicing": "unknown",
                "position_source": "fixture",
                "position_estimated": False,
                "before_lkfs": -20.0,
                "after_lkfs": -20.0,
                "flagged": False,
            })
        diagnostic = {
            "duration": .4,
            "sample_rate": 1000,
            "summary": {
                "join_count": 2, "flagged_join_count": 0,
                "exact_splice_count": 2, "estimated_splice_count": 0,
                "maximum_severity": 2.0,
            },
            "joins": joins,
            "segments": [
                {"phone": phone, "start": index * .1,
                 "end": (index + 1) * .1}
                for index, phone in enumerate(("a", "eh", "t", "pau"))
            ],
            "units": [],
            "join_curve": {},
            "momentary_curve": {},
        }
        dialog = fg.JoinDiscontinuityDialog(
            diagnostic, np.zeros(400, np.float32), focus_edge=2)
        try:
            selected = dialog.table_model.rows[
                dialog.table.currentIndex().row()]
            self.assertEqual(selected["segment_index"], 2)
            self.assertEqual(dialog.join_order.currentData(), "rendered")
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_view_menu_toggles_rendered_join_overlay(self):
        state = self._rendered_sentence_with_unit_choices("a b c")
        synthesis = state["synthesis"]
        synthesis.splice_records = [{
            "unit_index": 0,
            "segment_index": 1,
            "time": .15,
            "crossover_active": True,
            "crossover_start": .13,
            "crossover_end": .17,
        }]
        self.window.waveform.set_synthesis(synthesis)

        self.assertFalse(self.window.waveform.join_overlay_curve.isVisible())
        self.window.action_show_rendered_joins.setChecked(True)
        self.app.processEvents()

        self.assertTrue(self.window.waveform.join_overlay_curve.isVisible())
        self.assertIsNotNone(self.window.waveform._join_overlay_spans)
        self.assertTrue(self.window.cfg["show_rendered_joins"])

    def test_join_overlay_follows_phone_relative_timing_edits(self):
        state = self._rendered_sentence_with_unit_choices("a b c")
        synthesis = state["synthesis"]
        synthesis.splice_records = [{
            "unit_index": 0,
            "segment_index": 1,
            "time": .15,
            "crossover_active": True,
            "crossover_start": .13,
            "crossover_end": .17,
        }]
        waveform = self.window.waveform
        waveform.set_synthesis(synthesis)
        waveform.set_join_overlay_visible(True)

        before = waveform._join_overlay_geometry(
            synthesis.splice_records[0])
        waveform.set_factor(1, 2.0)
        after = waveform._join_overlay_geometry(
            synthesis.splice_records[0])

        self.assertAlmostEqual(before[0], .15)
        self.assertAlmostEqual(after[0], .20)
        self.assertAlmostEqual(after[1], .16)
        self.assertAlmostEqual(after[2], .24)

    def test_waveform_join_handles_show_and_obey_renderer_context_cap(self):
        state = self._rendered_sentence_with_unit_choices("a b c")
        synthesis = state["synthesis"]
        record = {
            "unit_index": 0,
            "segment_index": 1,
            "time": .15,
            "crossover_active": True,
            "crossover_start": .143,
            "crossover_end": .157,
            "crossover_effective_ms": 14.0,
            "crossover_context_cap_ms": 20.0,
        }
        synthesis.splice_records = [record]
        waveform = self.window.waveform
        waveform.set_synthesis(synthesis)
        waveform.set_requested_join_settings({
            "crossover_ms": 40.0,
            "crossover_overrides": {
                "0": {"left_ms": 10.0, "right_ms": 50.0},
            },
        })
        waveform.set_join_overlay_visible(True)
        waveform.set_join_overlay_editable(True)
        waveform._selected_join_record = 0
        waveform._redraw_join_overlays()

        requested = waveform._join_requested_edges(record, .15)
        self.assertAlmostEqual(
            (requested[2] - requested[1]) * 1000.0, 20.0)
        self.assertAlmostEqual(waveform._join_editor_max_seconds, .02)
        self.assertIn("cap 20.0 ms", waveform.join_edit_label.toPlainText())

    def test_waveform_join_focus_can_be_dismissed(self):
        class MarkerPoint:
            @staticmethod
            def data():
                return 0

        state = self._rendered_sentence_with_unit_choices("a b c")
        synthesis = state["synthesis"]
        synthesis.splice_records = [{
            "unit_index": 0,
            "segment_index": 1,
            "time": .15,
            "crossover_active": True,
            "crossover_start": .14,
            "crossover_end": .16,
            "crossover_effective_ms": 20.0,
            "crossover_context_cap_ms": 40.0,
        }]
        waveform = self.window.waveform
        waveform.set_synthesis(synthesis)
        waveform.set_join_overlay_visible(True)
        waveform.set_join_overlay_editable(True)

        waveform._join_marker_clicked(None, [MarkerPoint()], None)
        self.assertEqual(waveform._selected_join_record, 0)
        self.assertTrue(waveform.join_edit_label.isVisible())

        waveform._join_marker_clicked(None, [MarkerPoint()], None)
        self.assertIsNone(waveform._selected_join_record)
        self.assertFalse(waveform.join_edit_label.isVisible())
        self.assertFalse(waveform.join_left_handle.isVisible())
        self.assertFalse(waveform.join_right_handle.isVisible())

        waveform._join_marker_clicked(None, [MarkerPoint()], None)
        waveform._on_selection_drag(.15, .15, True)
        self.assertIsNone(waveform._selected_join_record)
        self.assertFalse(waveform.join_edit_label.isVisible())

    def test_waveform_join_handle_edit_marks_pending_and_is_undoable(self):
        state = self._rendered_sentence_with_unit_choices("a b c")
        state["engine"] = "festival_wsl"
        synthesis = state["synthesis"]
        synthesis.splice_records = [{
            "unit_index": 0,
            "segment_index": 1,
            "time": .15,
            "crossover_active": True,
            "crossover_start": .13,
            "crossover_end": .17,
        }]
        self.window.sentences = [state]
        self.window._active_sentence_index = 0
        self.window.current = synthesis
        self.window.waveform.set_synthesis(synthesis)
        with mock.patch.object(
                self.window, "_engine", return_value="festival_wsl"):
            self.window.action_show_rendered_joins.setChecked(True)
            self.window._refresh_join_overlay_controls()
            self.window._set_join_crossover_override(0, 10.0, 45.0)

        self.assertEqual(
            state["join_settings"]["crossover_overrides"]["0"],
            {"left_ms": 10.0, "right_ms": 45.0})
        self.assertTrue(state["needs_rerender"])
        self.assertEqual(self.window.waveform._pending_action, "rerender")
        self.assertEqual(
            self.window.waveform.curve.opts["pen"].color().name(),
            "#777777")
        self.assertTrue(self.window.waveform._join_overlay_editable)
        self.window.undo_stack.undo()
        self.assertEqual(state["join_settings"], {})
        self.assertFalse(state["needs_rerender"])
        self.assertEqual(self.window.waveform._pending_action, "")
        self.assertEqual(
            self.window.waveform.curve.opts["pen"].color().name(),
            "#1010c0")
        self.window.undo_stack.redo()
        self.assertEqual(
            state["join_settings"]["crossover_overrides"]["0"],
            {"left_ms": 10.0, "right_ms": 45.0})
        self.assertTrue(state["needs_rerender"])

    def test_join_window_controls_disable_under_legacy_fault(self):
        diagnostic = {
            "duration": .2, "sample_rate": 1000,
            "summary": {
                "join_count": 0, "flagged_join_count": 0,
                "exact_splice_count": 0, "estimated_splice_count": 0,
                "maximum_severity": 0.0,
            },
            "joins": [], "segments": [], "units": [],
            "join_curve": {}, "momentary_curve": {},
        }
        dialog = fg.JoinDiscontinuityDialog(
            diagnostic, np.zeros(200, np.float32),
            requested_join_settings={
                "mode": "asymmetric", "window_factor": 1.2},
            effective_join_settings={
                "window_symmetric": True, "window_factor": 1.0,
                "source": "legacy-fault"},
            editable=True, legacy_active=True)
        try:
            self.assertFalse(dialog.join_window_mode.isEnabled())
            self.assertFalse(dialog.join_window_factor.isEnabled())
            self.assertFalse(dialog.join_crossover_ms.isEnabled())
            self.assertFalse(dialog.apply_window_button.isEnabled())
            self.assertIn(
                "Legacy joins is active",
                dialog.join_window_explanation.text())
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_join_window_edit_preserves_units_timing_and_f0_and_undo(self):
        state = self._rendered_sentence_with_unit_choices("a b c")
        state["engine"] = "festival_wsl"
        synthesis = state["synthesis"]
        synthesis.targets = [(0.05, 120.0), (0.25, 115.0)]
        synthesis.generated_targets = list(synthesis.targets)
        selected_before = dict(synthesis.selected_units)
        overrides_before = dict(synthesis.unit_overrides)
        segment_before = [
            (segment.phone, segment.start, segment.end)
            for segment in synthesis.segments]
        targets_before = list(synthesis.targets)
        self.window.sentences = [state]
        self.window._active_sentence_index = 0
        self.window.current = synthesis
        self.window.waveform.set_synthesis(synthesis)

        self.window._set_join_settings({
            "mode": "asymmetric", "window_factor": 1.12})

        self.assertEqual(state["join_settings"], {
            "mode": "asymmetric", "window_factor": 1.12,
            "crossover_ms": 40.0, "crossover_overrides": {}})
        self.assertTrue(state["needs_rerender"])
        self.assertEqual(self.window.waveform._pending_action, "rerender")
        self.assertEqual(
            self.window.waveform.curve.opts["pen"].color().name(),
            "#777777")
        self.assertEqual(synthesis.selected_units, selected_before)
        self.assertEqual(synthesis.unit_overrides, overrides_before)
        self.assertEqual(
            [(segment.phone, segment.start, segment.end)
             for segment in synthesis.segments],
            segment_before)
        self.assertEqual(synthesis.targets, targets_before)
        self.window.undo_stack.undo()
        self.assertEqual(state["join_settings"], {})
        self.assertFalse(state["needs_rerender"])
        self.assertEqual(self.window.waveform._pending_action, "")
        self.window.undo_stack.redo()
        self.assertEqual(state["join_settings"], {
            "mode": "asymmetric", "window_factor": 1.12,
            "crossover_ms": 40.0, "crossover_overrides": {}})
        self.assertTrue(state["needs_rerender"])

    def test_project_round_trip_keeps_requested_and_effective_join_window(self):
        state = self._rendered_sentence_with_unit_choices("a b c")
        state["join_settings"] = {
            "mode": "symmetric", "window_factor": 1.07}
        state["synthesis"].join_settings = {
            "scope": "utterance",
            "requested_mode": "symmetric",
            "window_symmetric": True,
            "window_factor": 1.07,
            "source": "manual-window",
        }

        row = self.window._sentence_project_row(state)
        restored = self.window._state_from_project_row(row)

        self.assertEqual(restored["join_settings"], {
            "mode": "symmetric", "window_factor": 1.07,
            "crossover_ms": 40.0, "crossover_overrides": {}})
        self.assertEqual(
            restored["synthesis"].join_settings["source"],
            "manual-window")
        self.assertNotIn("_join_settings", restored["fault_mode"])

    def test_project_round_trip_keeps_asaxi_expression_provenance(self):
        state = self._rendered_sentence_with_unit_choices("ga vi")
        state["language"] = "Asaxi"
        state["lang_code"] = "asaxi"
        state["synthesis"].lang = "asaxi"
        state["synthesis"].asaxi_prosody = {
            "schema_version": 1,
            "dictionary_ruleset": "asaxi-pitch-v1",
            "word_count": 2,
            "phrases": [{
                "words": [{
                    "surface": "ga",
                    "phrase_expression": "ga vi",
                }, {
                    "surface": "vi",
                    "phrase_expression": "ga vi",
                }],
            }],
        }

        row = self.window._sentence_project_row(state)
        restored = self.window._state_from_project_row(row)

        self.assertEqual(
            restored["synthesis"].asaxi_prosody["word_count"], 2)
        self.assertEqual(
            restored["synthesis"].asaxi_prosody["phrases"][0]["words"][1][
                "phrase_expression"
            ],
            "ga vi",
        )

    def test_project_round_trip_keeps_structural_phone_source_view(self):
        state = self._rendered_sentence_with_unit_choices("i cl s o")
        state["synthesis"].phones = ["i", "cl", "s", "o"]
        state["synthesis"].segments = [
            fc.Segment("i", 0.0, 0.1),
            fc.Segment("cl", 0.1, 0.18),
            fc.Segment("s", 0.18, 0.28),
            fc.Segment("o", 0.28, 0.4),
        ]
        state["editor_segments"] = copy.deepcopy(
            state["synthesis"].segments)
        state["synthesis"].render_phones = ["i", "s", "s", "o"]
        state["synthesis"].special_phone_realizations = [{
            "index": 1,
            "phone": "cl",
            "mode": "anticipatory_consonant",
            "source_phone": "s",
            "status": "resolved",
        }]

        row = self.window._sentence_project_row(state)
        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "structural-phone-project"
            fc.save_project_folder(project, [row])
            normalized = fc.load_project(project)["sentences"][0]
        restored = self.window._state_from_project_row(normalized)

        self.assertEqual(
            restored["synthesis"].render_phones, ["i", "s", "s", "o"])
        self.assertEqual(
            restored["synthesis"].special_phone_realizations[0][
                "source_phone"],
            "s",
        )

    def test_sentences_follow_newly_spoken_row_only_when_enabled(self):
        self.window.sentences = [
            self.window._new_sentence_state(text)
            for text in ("one", "two", "three")]
        view = self.window.sentences_view
        view.refresh(self.window.sentences)
        with mock.patch.object(
                view.scroll, "ensureWidgetVisible") as ensure_visible:
            view.set_playing_item(0, 0)
            view.set_playing_item(0, 1)
            view.set_playing_item(2, 0)
            self.assertEqual(ensure_visible.call_count, 2)
            view.follow_spoken_sentence.setChecked(False)
            view.set_playing_item(1, 0)
            self.assertEqual(ensure_visible.call_count, 2)

    def test_phrase_snapshots_keep_variable_pauses_and_unmatched_tail(self):
        state = self.window.sentences[0]
        state["text"] = (
            "the synthesis commence. it had begun and it had ended.")
        phones = [
            "pau", "a", "b", "pau", "pau", "pau", "pau",
            "c", "d", "pau", "e", "f", "pau",
        ]
        segments = [
            fc.Segment(phone, index * .1, (index + 1) * .1)
            for index, phone in enumerate(phones)
        ]
        samples = np.linspace(-.4, .4, 1300, dtype=np.float32)
        synthesis = fc.Synthesis(
            samples.copy(), 1000, segments, text=state["text"])
        self.window.current = synthesis
        self.window.waveform.set_synthesis(synthesis)

        self.window._capture_phrase_snapshots(state)

        phrases = state["phrases"]
        self.assertEqual(len(phrases), 2)
        previews = state["phrase_previews"]
        chunks = [previews[phrase["id"]][0] for phrase in phrases]
        np.testing.assert_array_equal(np.concatenate(chunks), samples)
        self.assertEqual([len(chunk) for chunk in chunks], [500, 800])
        self.assertEqual(phrases[1]["phones"], ["c", "d", "e", "f"])
        self.assertAlmostEqual(phrases[0]["playback_end"], .5)
        self.assertAlmostEqual(phrases[1]["playback_start"], .5)

    def test_render_state_and_phrase_previews_share_audio_buffers(self):
        state = self.window.sentences[0]
        state["text"] = "rabbit."
        phones = ["pau", "r", "ae1", "b", "ih0", "t", "pau"]
        segments = [
            fc.Segment(phone, index * .1, (index + 1) * .1)
            for index, phone in enumerate(phones)
        ]
        samples = np.linspace(-.4, .4, 700, dtype=np.float32)
        synthesis = fc.Synthesis(
            samples, 1000, segments, text=state["text"], lang="en",
            phones=phones)

        self.window._commit_synthesis_to_state(
            state, synthesis, state["text"])

        self.assertIs(state["preview_audio"], samples)
        phrase = state["phrases"][0]
        preview = state["phrase_previews"][phrase["id"]][0]
        self.assertTrue(np.shares_memory(preview, samples))
        mini = fg.MiniWaveform((preview, 1000))
        self.assertTrue(np.shares_memory(mini.samples, preview))

    def test_sentence_snapshot_shares_audio_but_copies_editor_metadata(self):
        state = self._rendered_sentence_with_unit_choices("snapshot")
        state["phrase_previews"] = {
            "phrase": (state["preview_audio"][20:120], state["preview_sr"])
        }

        snapshot = self.window._sentence_state_snapshot(state)

        self.assertIs(snapshot["preview_audio"], state["preview_audio"])
        self.assertIs(
            snapshot["synthesis"].samples, state["synthesis"].samples)
        self.assertIs(
            snapshot["phrase_previews"]["phrase"][0],
            state["phrase_previews"]["phrase"][0])
        self.assertIsNot(snapshot["synthesis"], state["synthesis"])
        self.assertIsNot(
            snapshot["editor_segments"][0], state["editor_segments"][0])
        snapshot["editor_segments"][0].phone = "changed"
        self.assertEqual(state["editor_segments"][0].phone, "a")

    def test_curve_linguistic_unit_overlay_is_opt_in_and_persisted(self):
        self.assertFalse(fc.DEFAULT_CONFIG[
            "show_curve_linguistic_units"])
        self.assertFalse(self.window.curve_unit_overlay.isChecked())

        self.window.curve_unit_overlay.setChecked(True)

        self.assertTrue(
            self.window.cfg["show_curve_linguistic_units"])
        saved = fc.load_config(fg.CONFIG_PATH)
        self.assertTrue(saved["show_curve_linguistic_units"])

    def test_curve_unit_toggle_is_visible_only_for_continuous_curves(self):
        for mode in ("timing", "intonation", "recordings",
                     "japanese", "mora_voicing"):
            self.window.parameter_mode.setCurrentIndex(
                self.window.parameter_mode.findData(mode))
            self.window._on_parameter_mode()
            self.assertTrue(self.window.curve_unit_overlay.isHidden(), mode)
        for mode in ("pitch", "voicing", "vocal_tract"):
            self.window.parameter_mode.setCurrentIndex(
                self.window.parameter_mode.findData(mode))
            self.window._on_parameter_mode()
            self.assertFalse(self.window.curve_unit_overlay.isHidden(), mode)

    def test_english_syllables_overlay_all_continuous_curves_only(self):
        phones = ["pau", "r", "ae1", "b", "ih0", "t", "pau"]
        segments = [
            fc.Segment(phone, index * .1, (index + 1) * .1)
            for index, phone in enumerate(phones)
        ]
        synthesis = fc.Synthesis(
            np.zeros(700, np.float32), 1000, segments,
            text="rabbit", lang="en", phones=phones)
        self.window.current = synthesis
        self.window.waveform.set_synthesis(synthesis)
        self.window.text.setText("rabbit")

        self.window._sync_timing_track(reset=True)
        self.assertEqual(self.window.voicing_track._syllable_rows, [])
        self.window.curve_unit_overlay.setChecked(True)

        rows = self.window.voicing_track._syllable_rows
        self.assertEqual([row["label"] for row in rows],
                         ["r ae1", "b ih0 t"])
        self.assertAlmostEqual(rows[0]["start"], .1)
        self.assertAlmostEqual(rows[0]["end"], .3)
        self.assertAlmostEqual(rows[1]["start"], .3)
        self.assertAlmostEqual(rows[1]["end"], .6)
        for track in (
                self.window.pitch_track,
                self.window.voicing_track,
                self.window.vocal_tract_track):
            self.assertEqual(
                [row["label"] for row in track._syllable_rows],
                ["r ae1", "b ih0 t"],
            )

        self.window.waveform.segments[2].phone = "ih1"
        self.window._sync_timing_track(reset=True)
        self.assertEqual(
            synthesis.english_syllabification["phones"],
            ["pau", "r", "ih1", "b", "ih0", "t", "pau"],
        )

        synthesis.lang = "ja"
        self.window._sync_timing_track(reset=True)
        self.assertEqual(self.window.voicing_track._syllable_rows, [])

    def test_english_inline_multilingual_vowel_splits_overlay(self):
        phones = ["dh", "ih", "s", "ih", "z", "a", "t", "eh", "s", "t"]
        segments = [
            fc.Segment(phone, index * .1, (index + 1) * .1)
            for index, phone in enumerate(phones)
        ]
        synthesis = fc.Synthesis(
            np.zeros(1000, np.float32), 1000, segments,
            text="[dh ih s ih z a t eh s t]", lang="en", phones=phones)
        self.window.current = synthesis
        self.window.waveform.set_synthesis(synthesis)
        self.window.text.setText(synthesis.text)
        self.window.curve_unit_overlay.setChecked(True)

        self.window._sync_timing_track(reset=True)

        expected = ["dh ih", "s ih", "z a", "t eh s t"]
        for track in (
                self.window.pitch_track,
                self.window.voicing_track,
                self.window.vocal_tract_track):
            self.assertEqual(
                [row["label"] for row in track._syllable_rows],
                expected,
            )

    def test_japanese_moras_overlay_all_continuous_curves(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "\u304b\u306a", mode="kana")
        plan = fg.je.create_edited_plan(
            utterance, fg.je.new_edit_state(utterance),
            runtime_metadata={"language": "ja"})
        segments = fc.segments_from_durations(plan.segment_durations)
        duration = segments[-1].end
        synthesis = fc.Synthesis(
            np.zeros(max(1, int(duration * 1000)), np.float32), 1000,
            segments, text=utterance.source_text, lang="ja")
        state = self.window.sentences[0]
        overlay = fg.je.new_edit_state(utterance, frontend_mode="kana")
        overlay["last_plan"] = plan.to_dict()
        state["lang_code"] = "ja"
        state["japanese_state"] = overlay
        state["synthesis"] = synthesis
        state["rendered"] = True
        self.window.current = synthesis
        self.window.waveform.set_synthesis(synthesis)
        self.window.japanese_editor.set_state(overlay)
        self.window.curve_unit_overlay.setChecked(True)

        self.window._sync_timing_track(reset=True)

        expected = [
            mora.surface or mora.reading for mora in utterance.moras
        ]
        for track in (
                self.window.pitch_track,
                self.window.voicing_track,
                self.window.vocal_tract_track):
            rows = track._syllable_rows
            self.assertEqual([row["label"] for row in rows], expected)
            self.assertTrue(all(row["kind"] == "mora" for row in rows))
            self.assertTrue(all(
                row["tooltip"].startswith("Japanese mora")
                for row in rows
            ))

    def test_asaxi_moras_overlay_all_continuous_curves(self):
        phones = ["pau", "sh", "er", "s", "o", "pau"]
        segments = [
            fc.Segment(phone, index * .1, (index + 1) * .1)
            for index, phone in enumerate(phones)
        ]
        synthesis = fc.Synthesis(
            np.zeros(600, np.float32), 1000, segments,
            text="sh\u011bso", lang="asaxi", phones=phones)
        metadata = {
            "rendered_phones": ["sh", "er", "s", "o"],
            "moras": [
                {
                    "mora_index": 0,
                    "phrase_index": 0,
                    "word": "sh\u011bso",
                    "text": "sh\u011b",
                    "phones": ["sh", "er"],
                    "pitch": "H",
                    "segment_indices": [0, 1],
                },
                {
                    "mora_index": 1,
                    "phrase_index": 0,
                    "word": "sh\u011bso",
                    "text": "so",
                    "phones": ["s", "o"],
                    "pitch": "L",
                    "segment_indices": [2, 3],
                },
            ],
        }
        state = self.window.sentences[0]
        overlay = fg.asaxi_editing.new_edit_state("sh\u011bso")
        overlay["last_plan"] = metadata
        state["lang_code"] = "asaxi"
        state["asaxi_state"] = overlay
        state["synthesis"] = synthesis
        state["rendered"] = True
        self.window.current = synthesis
        self.window.waveform.set_synthesis(synthesis)
        self.window.curve_unit_overlay.setChecked(True)

        self.window._sync_timing_track(reset=True)

        rows = self.window.voicing_track._syllable_rows
        self.assertEqual([row["label"] for row in rows],
                         ["sh\u011b", "so"])
        self.assertAlmostEqual(rows[0]["start"], 0.1)
        self.assertAlmostEqual(rows[0]["end"], 0.3)
        self.assertAlmostEqual(rows[1]["start"], 0.3)
        self.assertAlmostEqual(rows[1]["end"], 0.5)
        self.assertTrue(all(
            row["tooltip"].startswith("Asaxi mora")
            for row in rows
        ))
        for track in (
                self.window.pitch_track,
                self.window.vocal_tract_track):
            self.assertEqual(
                [row["label"] for row in track._syllable_rows],
                ["sh\u011b", "so"],
            )

    def test_project_row_preserves_english_syllable_metadata(self):
        state = self._rendered_sentence_with_unit_choices("rabbit")
        state["lang_code"] = "en"
        state["synthesis"].lang = "en"
        state["synthesis"].phones = ["r", "ae1", "b", "ih0", "t"]
        state["synthesis"].english_syllabification = (
            fc.english_syllable_domain.syllabify_english(
                state["synthesis"].phones).to_dict())

        row = self.window._sentence_project_row(state)
        restored = self.window._state_from_project_row(row)

        self.assertEqual(
            restored["synthesis"].english_syllabification,
            state["synthesis"].english_syllabification,
        )

    def test_single_explicit_pause_keeps_one_sentence_phrase_preview(self):
        state = self.window.sentences[0]
        state["text"] = "one [pau] continuous phrase"
        phones = ["pau", "pau", "w", "ah", "n", "pau", "k", "ey",
                  "pau", "pau"]
        segments = [
            fc.Segment(phone, index * .1, (index + 1) * .1)
            for index, phone in enumerate(phones)
        ]
        samples = np.linspace(-.4, .4, 1000, dtype=np.float32)
        synthesis = fc.Synthesis(
            samples.copy(), 1000, segments, text=state["text"])
        self.window.current = synthesis
        self.window.waveform.set_synthesis(synthesis)

        self.window._capture_phrase_snapshots(state)

        self.assertEqual(len(state["phrases"]), 1)
        phrase = state["phrases"][0]
        self.assertEqual(phrase["text"], state["text"])
        preview = state["phrase_previews"][phrase["id"]][0]
        np.testing.assert_array_equal(preview, samples)
        self.assertEqual(phrase["phones"], ["w", "ah", "n", "k", "ey"])

    def test_play_all_uses_complete_sentence_audio_not_phrase_previews(self):
        first = np.concatenate([
            np.full(10, .1, np.float32),
            np.zeros(3, np.float32),
            np.full(10, .2, np.float32),
            np.zeros(1, np.float32),
            np.full(10, .3, np.float32),
        ])
        second = np.full(20, -.2, np.float32)
        states = []
        for text, samples in (
                ("one. two and three.", first),
                ("\u4f55\u304c\u5909\u308f\u3063\u305f\u306e\u304b\u77e5\u3089\u306a\u3044\u304b\u3089\u6016\u3044\u3093\u3060\u3002", second)):
            state = self.window._new_sentence_state(text)
            state["rendered"] = True
            state["preview_audio"] = samples.copy()
            state["preview_sr"] = 100
            # Deliberately incomplete phrase previews must never truncate
            # sentence-level Play All.
            phrase = self.window._new_phrase_state(text)
            state["phrases"] = [phrase]
            state["phrase_previews"] = {
                phrase["id"]: (samples[:10].copy(), 100)}
            states.append(state)
        self.window.sentences = states
        self.window.sentences_view.refresh(states)
        self.window.sentences_view.set_selected_phrase_keys([(0, 0)])
        self.assertEqual(
            self.window.sentences_view.selected_phrase_keys(), [(0, 0)])
        self.window.sentences_view.set_selected_indices([])
        self.assertEqual(
            self.window.sentences_view.selected_phrase_keys(), [])
        self.assertEqual(self.window.sentences_view.play_all.text(),
                         "Play all")
        played = {}

        with mock.patch.object(self.window, "_capture_active_sentence"), \
                mock.patch.object(
                    self.window, "_start_playback",
                    side_effect=lambda samples, sr, **kwargs: played.update(
                        samples=np.asarray(samples).copy(), sr=sr,
                        highlights=kwargs.get("highlights"))):
            self.window._play_all_sentences()

        expected, expected_sr = fc.concat_audio(
            [(first, 100), (second, 100)], gap_s=.25)
        np.testing.assert_array_equal(played["samples"], expected)
        self.assertEqual(played["sr"], expected_sr)
        self.assertTrue(np.any(played["samples"] == .3))

    def test_generate_all_populates_every_sentence_waveform_and_keeps_tab(self):
        self.window.sentences = [
            self.window._new_sentence_state(text)
            for text in ("one.", "two.", "three.")]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)
        self.window.mode_tabs.setCurrentIndex(1)

        active_indices = []

        def generate(*_args, **kwargs):
            state = kwargs["target_state"]
            index = self.window.sentences.index(state)
            active_indices.append(self.window._active_sentence_index)
            duration = 0.3 + index * 0.05
            samples = np.ones(int(duration * 1000), np.float32) * 0.1
            synthesis = fc.Synthesis(
                samples,
                1000,
                [fc.Segment("pau", 0.0, 0.05),
                 fc.Segment("aa", 0.05, duration - 0.05),
                 fc.Segment("pau", duration - 0.05, duration)],
                text=self.window.sentences[index]["text"],
                voicebank="fixture",
            )
            self.window._commit_synthesis_to_state(
                state, synthesis, state["text"])
            self.window._last_generation_error = ""
            return synthesis

        with mock.patch.object(self.window, "_need_backend", return_value=True), \
                mock.patch.object(self.window, "_confirm_generate_reset",
                                  return_value=True), \
                mock.patch.object(self.window, "_generate_for_sentence_mode",
                                  side_effect=generate):
            self.window.on_generate_all()

        self.assertEqual(self.window.mode_tabs.currentIndex(), 1)
        self.assertEqual(self.window._active_sentence_index, 0)
        self.assertEqual(active_indices, [0, 0, 0])
        for state in self.window.sentences:
            self.assertGreater(np.asarray(state["preview_audio"]).size, 1)
            self.assertTrue(state.get("phrase_previews"))
            self.assertTrue(all(
                phrase["id"] in state["phrase_previews"]
                for phrase in state["phrases"]))

    def test_generate_all_has_visible_progress_and_manual_stop(self):
        self.window.sentences = [
            self.window._new_sentence_state(text)
            for text in ("one.", "two.", "three.")]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)
        observations = []

        def generate(*_args, **kwargs):
            observations.append((
                not self.window.batch_progress.isHidden(),
                self.window.btn_stop.isEnabled(),
            ))
            state = kwargs["target_state"]
            synthesis = fc.Synthesis(
                np.ones(300, np.float32) * 0.1,
                1000,
                [fc.Segment("pau", 0.0, 0.05),
                 fc.Segment("aa", 0.05, 0.25),
                 fc.Segment("pau", 0.25, 0.3)],
                text=state["text"],
                voicebank="fixture",
            )
            self.window._commit_synthesis_to_state(
                state, synthesis, state["text"])
            self.window.on_stop()
            return synthesis

        with mock.patch.object(self.window, "_need_backend", return_value=True), \
                mock.patch.object(self.window, "_confirm_generate_reset",
                                  return_value=True), \
                mock.patch.object(self.window, "_generate_for_sentence_mode",
                                  side_effect=generate):
            self.window.on_generate_all()

        self.assertEqual(observations, [(True, True)])
        self.assertTrue(self.window.sentences[0]["rendered"])
        self.assertFalse(self.window.sentences[1]["rendered"])
        self.assertTrue(self.window.batch_progress.isHidden())
        self.assertTrue(self.window.batch_cancel.isHidden())
        self.assertIn("stopped after 1", self.window.statusBar().currentMessage())

    def test_generate_all_groups_identical_failures_into_one_dialog(self):
        self.window.sentences = [
            self.window._new_sentence_state(text)
            for text in ("one", "two", "three")]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)

        def fail(*_args, **_kwargs):
            self.window._last_generation_error = (
                "Festival (WSL) failed (exit 11):\n(no output)")
            return None

        with mock.patch.object(self.window, "_need_backend", return_value=True), \
                mock.patch.object(self.window, "_confirm_generate_reset",
                                  return_value=True), \
                mock.patch.object(self.window, "_generate_for_sentence_mode",
                                  side_effect=fail), \
                mock.patch.object(fg.QtWidgets.QMessageBox, "warning") as warning, \
                mock.patch.object(fg.QtWidgets.QMessageBox, "critical") as critical:
            self.window.on_generate_all()

        self.assertEqual(warning.call_count, 1)
        self.assertEqual(critical.call_count, 0)
        self.assertIn("Sentences 1, 2, 3", warning.call_args.args[2])
        self.assertEqual(
            warning.call_args.args[2].count("Festival (WSL) failed"), 1)

    def test_sentences_generate_shortcut_targets_selection_without_playback(self):
        self.window.sentences = [
            self.window._new_sentence_state(text)
            for text in ("one", "two", "three")]
        self.window._refresh_sentence_selector(0)
        self.window._refresh_sentences_view()
        self.window.mode_tabs.setCurrentIndex(1)
        self.window.sentences_view.set_selected_indices([0, 1, 2])

        with mock.patch.object(self.window, "on_generate_all") as generate, \
                mock.patch.object(self.window, "on_generate") as current, \
                mock.patch.object(
                    self.window, "_play_sentence_indices") as playback:
            self.window._dispatch_shortcut("generate")

        generate.assert_called_once_with(only_indices=[0, 1, 2])
        current.assert_not_called()
        playback.assert_not_called()

    def test_generate_all_never_follows_target_and_keeps_user_navigation(self):
        self.window.sentences = [
            self.window._new_sentence_state(text)
            for text in ("one.", "two.", "three.")]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)
        self.window.mode_tabs.setCurrentIndex(1)
        self.window.sentences_view.set_selected_indices([0])
        observations = []

        def generate(*_args, **kwargs):
            target = kwargs["target_state"]
            target_index = self.window.sentences.index(target)
            if target_index == 0:
                self.window.sentences_view.set_selected_indices([2])
                self.window.mode_tabs.setCurrentIndex(0)
            observations.append((
                target_index, self.window._active_sentence_index,
                self.window.mode_tabs.currentIndex()))
            synthesis = fc.Synthesis(
                np.ones(200, np.float32) * .1, 1000,
                [fc.Segment("pau", 0, .05),
                 fc.Segment("aa", .05, .15),
                 fc.Segment("pau", .15, .2)],
                text=target["text"], voicebank="fixture")
            self.window._commit_synthesis_to_state(
                target, synthesis, target["text"])
            return synthesis

        with mock.patch.object(self.window, "_need_backend", return_value=True), \
                mock.patch.object(self.window, "_confirm_generate_reset",
                                  return_value=True), \
                mock.patch.object(self.window, "_generate_for_sentence_mode",
                                  side_effect=generate):
            self.window.on_generate_all()

        self.assertEqual(observations, [
            (0, 2, 0), (1, 2, 0), (2, 2, 0)])
        self.assertEqual(self.window._active_sentence_index, 2)
        self.assertEqual(self.window.sentence_select.currentIndex(), 2)
        self.assertEqual(self.window.mode_tabs.currentIndex(), 0)

    def test_sentence_multiselect_group_move_and_undo(self):
        self.window.sentences = [
            self.window._new_sentence_state(text)
            for text in ("A", "B", "C", "D")]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)
        self.window._refresh_sentences_view()
        view = self.window.sentences_view

        view._select_sentence(0)
        view._select_sentence(2, fg.Qt.ControlModifier)
        view._select_sentence(
            1, fg.Qt.ControlModifier | fg.Qt.ShiftModifier)
        self.assertEqual(view.selected_sentence_indices(), [0, 1, 2])

        self.window._move_sentence_group([0, 2], 4)
        self.assertEqual([state["text"] for state in self.window.sentences],
                         ["B", "D", "A", "C"])
        self.assertEqual(view.selected_sentence_indices(), [2, 3])
        self.window.undo_stack.undo()
        self.assertEqual([state["text"] for state in self.window.sentences],
                         ["A", "B", "C", "D"])
        self.window.undo_stack.redo()
        self.assertEqual([state["text"] for state in self.window.sentences],
                         ["B", "D", "A", "C"])

    def test_multiselected_sentence_sidebar_shows_mixed_and_bulk_applies(self):
        class FakeVoices:
            @staticmethod
            def voicebanks():
                return [
                    {"name": "voice_a", "dir": "a", "ok": True,
                     "source": "test"},
                    {"name": "voice_b", "dir": "b", "ok": True,
                     "source": "test"},
                ]

            @staticmethod
            def default_voicebank():
                return "voice_a"

            @staticmethod
            def voice_pitch_hz(_voice):
                return None

        self.window.backend = FakeVoices()
        self.window.sentences = [
            self.window._new_sentence_state(text)
            for text in ("A", "B", "C")]
        self.window.sentences[0].update({
            "language": "English", "lang_code": "en",
            "voicebank": "voice_a"})
        self.window.sentences[1].update({
            "language": "Japanese", "lang_code": "ja",
            "voicebank": "voice_b"})
        self.window.sentences[2].update({
            "language": "Asaxi", "lang_code": "asaxi",
            "voicebank": "voice_a"})
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)
        self.window.mode_tabs.setCurrentIndex(1)
        self.window.sentences_view.set_selected_indices([0, 1])
        self.app.processEvents()

        self.assertEqual(
            self.window.lang.currentData(), fg.MIXED_SELECTION_DATA)
        self.assertEqual(
            self.window.voicebank.currentItem().data(fg.Qt.UserRole),
            fg.MIXED_SELECTION_DATA)

        self.window.lang.setCurrentText("English")
        self.assertTrue(self.window._select_voicebank_name("voice_b"))
        self.app.processEvents()

        for state in self.window.sentences[:2]:
            self.assertEqual(state["language"], "English")
            self.assertEqual(state["lang_code"], "en")
            self.assertEqual(state["voicebank"], "voice_b")
            self.assertEqual(self.window._pending_action(state), "generate")
        self.assertEqual(
            self.window.sentences[0]["pending_reason"], "Voicebank changed")
        self.assertEqual(self.window.sentences[2]["language"], "Asaxi")
        self.assertEqual(self.window.sentences[2]["voicebank"], "voice_a")

        self.window.mode_tabs.setCurrentIndex(0)
        self.assertEqual(
            self.window.sentences_view.selected_sentence_indices(), [])
        self.assertEqual(self.window.sentences_view.selected_phrase_keys(), [])

    def test_selected_sentence_language_is_editable_before_speaker_change(self):
        compatibility = fc.VoiceCompatibility(
            metadata_status="current",
            primary_language="ja",
            supported_languages=("ja",),
            voice_entry_points={"ja": "voice_fixture_ja"},
        )
        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        self.window.voicebank.clear()
        item = fg.QtWidgets.QListWidgetItem("fixture")
        item.setData(fg.Qt.UserRole, "fixture")
        self.window.voicebank.addItem(item)
        self.window.voicebank.setCurrentItem(item)
        state = self.window.sentences[0]
        state.update({
            "engine": "festival_wsl",
            "language": "Japanese",
            "lang_code": "ja",
            "voicebank": "fixture",
        })
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)

        with mock.patch.object(
                self.window.fest, "voice_compatibility",
                return_value=compatibility):
            self.window.mode_tabs.setCurrentIndex(1)
            self.window.sentences_view.set_selected_indices([0])
            self.window._apply_voice_language_compatibility(
                auto_select=False)
            english = self.window.lang.findText("English")
            self.assertGreaterEqual(english, 0)
            self.assertTrue(
                self.window.lang.model().item(english).isEnabled())

            self.window.lang.setCurrentIndex(english)
            self.app.processEvents()

        self.assertEqual(state["language"], "English")
        self.assertEqual(state["lang_code"], "en")
        self.assertEqual(state["voicebank"], "fixture")
        self.assertEqual(self.window._pending_action(state), "generate")
        self.assertEqual(
            self.window.sentences_view.selected_sentence_indices(), [0])
        self.assertTrue(self.window.sidebar_editor.isEnabled())

    def test_multilingual_voice_preserves_supported_mixed_languages(self):
        class FakeMultilingualVoice:
            @staticmethod
            def voice_compatibility(_voice):
                return fc.VoiceCompatibility(
                    metadata_status="current",
                    primary_language="en",
                    supported_languages=("en", "ja"),
                    voice_entry_points={
                        "en": "voice_multi_en", "ja": "voice_multi_ja"},
                )

        self.window.fest = FakeMultilingualVoice()
        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        self.window.sentences = [
            self.window._new_sentence_state("English"),
            self.window._new_sentence_state("Japanese"),
        ]
        self.window.sentences[0].update({
            "language": "English", "lang_code": "en",
            "voicebank": "voice_multi",
        })
        self.window.sentences[1].update({
            "language": "Japanese", "lang_code": "ja",
            "voicebank": "voice_multi",
        })
        self.window.voicebank.clear()
        voice = fg.QtWidgets.QListWidgetItem("voice_multi")
        voice.setData(fg.Qt.UserRole, "voice_multi")
        self.window.voicebank.addItem(voice)
        self.window.voicebank.setCurrentItem(voice)
        self.window.mode_tabs.blockSignals(True)
        self.window.mode_tabs.setCurrentIndex(1)
        self.window.mode_tabs.blockSignals(False)
        self.window.sentences_view.refresh(self.window.sentences)
        self.window.sentences_view.set_selected_indices([0, 1])
        self.window._sync_sentence_sidebar_values([0, 1])

        self.window._apply_voice_language_compatibility(auto_select=True)

        self.assertEqual(
            self.window.lang.currentData(), fg.MIXED_SELECTION_DATA)
        self.assertEqual(
            [state["lang_code"] for state in self.window.sentences],
            ["en", "ja"],
        )

    def test_voicebank_manager_multiselect_and_batch_removal(self):
        class FakeVoices:
            def __init__(self):
                self.removed = []

            @staticmethod
            def voicebanks():
                return [
                    {"name": "voice_a", "dir": "a", "ok": True,
                     "source": "test"},
                    {"name": "voice_b", "dir": "b", "ok": True,
                     "source": "test"},
                ]

            @staticmethod
            def default_voicebank():
                return "voice_a"

            @staticmethod
            def voice_pitch_hz(_voice):
                return None

            @staticmethod
            def voicebank_removal_info(name):
                return {"name": name, "path": "missing/" + name,
                        "kind": "windows", "exists": False}

            def uninstall_voicebank(self, name, delete_files=True):
                self.removed.append((name, delete_files))
                return "missing/" + name

        backend = FakeVoices()
        self.window.backend = backend
        self.window.cfg["extra_voicebanks"] = {
            "voice_a": "a", "voice_b": "b"}
        captured = {}

        def inspect_manager(dialog):
            table = dialog.findChild(fg.QtWidgets.QTreeWidget)
            captured["mode"] = table.selectionMode()
            table.topLevelItem(0).setSelected(True)
            table.topLevelItem(1).setSelected(True)
            self.app.processEvents()
            captured["selected"] = len(table.selectedItems())
            captured["delete"] = next(
                button.text() for button in dialog.findChildren(
                    fg.QtWidgets.QPushButton)
                if button.text().startswith("Delete"))
            next(
                button for button in dialog.findChildren(
                    fg.QtWidgets.QPushButton)
                if button.text().startswith("Delete")).click()
            return fg.QtWidgets.QDialog.Rejected

        def accept_confirmation(box):
            button = next(
                item for item in box.buttons()
                if box.buttonRole(item) in {
                    fg.QtWidgets.QMessageBox.AcceptRole,
                    fg.QtWidgets.QMessageBox.DestructiveRole})
            button.click()
            return 0

        with mock.patch.object(
                fg.QtWidgets.QMessageBox, "exec_", accept_confirmation), \
                mock.patch.object(
                    fg.QtWidgets.QDialog, "exec_", inspect_manager):
            self.window.on_voicebank_manager()
        self.assertEqual(
            captured["mode"],
            fg.QtWidgets.QAbstractItemView.ExtendedSelection)
        self.assertEqual(captured["selected"], 2)
        self.assertEqual(captured["delete"], "Delete 2...")
        self.assertEqual(backend.removed, [
            ("voice_a", False), ("voice_b", False)])

    def test_recordings_exports_named_broadband_impulse_audit(self):
        segments = [fc.Segment("a", 0.0, 0.2),
                    fc.Segment("k", 0.2, 0.4),
                    fc.Segment("a", 0.4, 0.6)]
        synthesis = fc.Synthesis(
            np.zeros(600, np.float32), 1000, segments,
            selected_units={0: "a", 1: "k"})
        self.window._show_synthesis(synthesis)
        diagnostic = {
            "summary": {"flagged_join_count": 0},
            "segments": [
                {"phone": row.phone, "start": row.start, "end": row.end}
                for row in segments
            ],
            "joins": [],
        }

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)

            def render(_samples, _rate, destination, **_kwargs):
                destination = Path(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"PNG")
                return destination

            with mock.patch.object(fg, "FESTVOX_TOOL_DIR", root), \
                    mock.patch.object(
                        fg.diphone_loudness, "analyze_rendered_joins",
                        return_value=diagnostic), \
                    mock.patch.object(
                        fg.join_spectrogram, "render_join_spectrogram",
                        side_effect=render) as renderer, \
                    mock.patch.object(
                        fg.QtWidgets.QMessageBox, "information"):
                output = self.window.on_export_broadband_impulse_join_audit()

            self.assertEqual(output.parent.name,
                             "broadband_impulse_join_audit")
            self.assertEqual(output.parent.parent.name, "diagnostic_images")
            self.assertIn("broadband_impulse_join_audit", output.name)
            self.assertTrue(output.is_file())
            self.assertEqual(renderer.call_count, 1)
        self.assertIs(
            self.window.recordings_broadband_audit.parent(),
            self.window.recordings_page)
        self.assertIn("Broadband Impulse Join Audit",
                      self.window.recordings_broadband_audit.text())

    def test_showing_a_render_does_not_generate_diagnostic_graphs(self):
        samples = np.zeros(1600, np.float32)
        synthesis = fc.Synthesis(
            samples, 16000, [fc.Segment("a", 0.0, 0.1)])

        with mock.patch.object(
                fg.diphone_loudness, "analyze_rendered_joins") as joins, \
                mock.patch.object(
                    fg.rendered_formant_diagnostic,
                    "analyze_rendered_formants") as formants, \
                mock.patch.object(
                    fg.join_spectrogram,
                    "render_join_spectrogram") as spectrogram:
            self.window._show_synthesis(synthesis)

        joins.assert_not_called()
        formants.assert_not_called()
        spectrogram.assert_not_called()

    def test_rendered_formant_dialog_analyzes_final_waveform(self):
        sample_rate = 16000
        time = np.arange(6400, dtype=np.float64) / sample_rate
        samples = np.asarray(.2 * np.sin(2 * np.pi * 170.0 * time),
                             np.float32)
        segments = [fc.Segment("e", 0.0, 0.2),
                    fc.Segment("e", 0.2, 0.4)]
        synthesis = fc.Synthesis(samples, sample_rate, segments)
        self.window._show_synthesis(synthesis)
        report = {
            "kind": "rendered_formant_diagnostic",
            "duration_seconds": .4,
            "sample_rate": sample_rate,
            "accepted_frame_count": 2,
            "rejected_frame_count": 0,
            "analyzed_phone_count": 2,
            "potential_jump_count": 0,
            "phones": [],
            "jumps": [],
        }
        with mock.patch.object(
                fg.diphone_loudness, "analyze_rendered_joins",
                return_value={"joins": []}), mock.patch.object(
                fg.rendered_formant_diagnostic,
                "analyze_rendered_formants", return_value=report
                ) as analyzer, mock.patch.object(
                    fg, "RenderedFormantDialog") as dialog_class:
            result = self.window.on_rendered_formant_diagnostic()
        self.assertIs(result, report)
        self.assertEqual(analyzer.call_count, 1)
        np.testing.assert_array_equal(analyzer.call_args.args[0], samples)
        self.assertEqual(analyzer.call_args.args[1], sample_rate)
        self.assertEqual(list(analyzer.call_args.args[2]), segments)
        dialog_class.return_value.exec_.assert_called_once_with()

    def test_rendered_formant_dialog_draws_tracks_and_ranked_marker(self):
        sample_rate = 16000
        time = np.arange(4800, dtype=np.float64) / sample_rate
        samples = np.asarray(.2 * np.sin(2 * np.pi * 170.0 * time),
                             np.float32)
        frames = [
            {"time": .08, "formants_hz": [500, 1800, 2800, 3700]},
            {"time": .12, "formants_hz": [520, 1840, 2820, 3720]},
        ]
        report = {
            "duration_seconds": .3,
            "sample_rate": sample_rate,
            "accepted_frame_count": 2,
            "rejected_frame_count": 0,
            "analyzed_phone_count": 1,
            "potential_jump_count": 1,
            "analysis_frame_step_seconds": .01,
            "phones": [{"phone": "e", "start": 0.0, "end": .3,
                        "frames": frames}],
            "jumps": [{
                "rank": 1, "time": .15,
                "kind": "EXACT_SPLICE_FORMANT_JUMP",
                "left_phone": "e", "right_phone": "e",
                "severity": 2.0, "max_delta_cents": 480.0,
                "novelty": 4.5, "exact_splice_evidence": True,
                "interpretation": "potential exact splice jump",
            }],
        }
        dialog = fg.RenderedFormantDialog(report, samples)
        try:
            self.app.processEvents()
            self.assertEqual(dialog.table.rowCount(), 1)
            self.assertEqual(dialog.table.item(0, 2).text(), "e -> e")
            self.assertTrue(hasattr(dialog, "spectrogram_item"))
            self.assertGreater(len(dialog.formant_plot.listDataItems()), 0)
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_sentences_alt_selects_phrases_across_rows_and_add_inserts_after(self):
        self.window.sentences = [
            self.window._new_sentence_state(text) for text in ("A", "B")]
        for state, prefix in zip(self.window.sentences, ("a", "b")):
            state["phrases"] = [
                self.window._new_phrase_state(prefix + str(index))
                for index in range(2)]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._refresh_sentences_view()
        view = self.window.sentences_view

        view._select_phrase(0, 0)
        view._select_phrase(1, 1, fg.Qt.AltModifier)
        self.assertEqual(view.selected_phrase_keys(), [(0, 0), (1, 1)])
        self.assertEqual(view.selected_sentence_indices(), [0, 1])

        view.set_selected_indices([0])
        self.window._add_sentence_from_sentences_view([0])
        self.assertEqual(len(self.window.sentences), 3)
        self.assertEqual(self.window.sentences[2]["text"], "B")
        buttons = [button.text() for button in
                   view.findChildren(fg.QtWidgets.QPushButton)]
        self.assertNotIn("Split at playhead", buttons)

    def test_sentence_text_and_speaker_controls_are_contextual(self):
        self.window.sentences = [self.window._new_sentence_state("old")]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._refresh_sentences_view()
        self.window.mode_tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertFalse(self.window.sidebar_editor.isEnabled())
        self.assertFalse(self.window.sentences_view.selection_notice.isHidden())

        self.window.sentences_view.set_selected_indices([0])
        self.app.processEvents()
        self.assertTrue(self.window.sidebar_editor.isEnabled())
        row = self.window.sentences_view.row_widgets[0]
        editor = row.findChild(fg.SentenceTextEdit)
        editor.setPlainText("new sentence text")
        self.app.processEvents()
        self.assertEqual(self.window.sentences[0]["text"],
                         "new sentence text")
        self.assertFalse(self.window.sentences[0]["rendered"])
        self.assertEqual(len(row.findChildren(fg.SpeakerBadge)), 1)
        self.assertTrue(row.findChild(fg.ClickableLabel).toolTip())

    def test_switching_from_sentences_restores_waveform_and_parameters(self):
        segments = [fc.Segment("pau", 0, .1),
                    fc.Segment("a", .1, .3),
                    fc.Segment("pau", .3, .4)]
        syn = fc.Synthesis(
            np.ones(400, np.float32) * .1, 1000, segments,
            text="a", voicebank="voice", generated_targets=[(.2, 180.0)])
        self.window.mode_tabs.setCurrentIndex(1)
        state = self.window.sentences[0]
        state.update({"synthesis": syn, "rendered": True,
                      "preview_audio": syn.samples.copy(),
                      "preview_sr": syn.sr})
        self.window.sentences_view.set_selected_indices([0])
        self.window.current = None
        self.window.waveform.set_synthesis(fc.Synthesis(
            np.zeros(1, np.float32), 1000, []))

        self.window.mode_tabs.setCurrentIndex(0)
        self.app.processEvents()

        self.assertEqual(len(self.window.waveform.segments), 3)
        self.assertEqual(len(self.window.waveform.boundaries), 2)
        self.assertTrue(self.window.pitch_track.targets())

    def test_edit_background_generated_sentence_preserves_parameter_state(self):
        def generated_state(text, pitch):
            segments = [
                fc.Segment("pau", 0.0, 0.08),
                fc.Segment("a", 0.08, 0.26),
                fc.Segment("pau", 0.26, 0.34),
            ]
            syn = fc.Synthesis(
                np.linspace(-0.1, 0.1, 340, dtype=np.float32),
                1000,
                segments,
                text=text,
                voicebank="voice",
                generated_targets=[(0.08, pitch), (0.26, pitch - 8.0)],
                generated_voicing_targets=[
                    (0.0, 0.0), (0.08, 0.9), (0.26, 0.8), (0.34, 0.0)],
                generated_vocal_tract_targets=[
                    (0.0, 1.0), (0.17, 0.96), (0.34, 1.0)],
            )
            state = self.window._new_sentence_state(text)
            self.window._commit_synthesis_to_state(
                state, syn, text, timing_factors=[1.0, 1.15, 1.0])
            return state

        first = generated_state("first", 190.0)
        second = generated_state("second", 175.0)
        first_syn = first["synthesis"]
        self.window.sentences = [first, second]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._refresh_sentences_view()
        self.window.mode_tabs.setCurrentIndex(1)
        self.app.processEvents()

        # Model Generate All while Speech was never hydrated: the visible
        # editor is blank even though both sentence records contain renders.
        self.window.current = None
        self.window._editor_sentence_state = None
        self.window.waveform.set_synthesis(fc.Synthesis(
            np.zeros(1, np.float32), 1000, []))

        # Selecting another row used to capture the blank editor into the
        # first sentence and erase its synthesis.
        self.window.sentences_view.set_selected_indices([1])
        self.app.processEvents()
        self.assertIs(first["synthesis"], first_syn)

        row = self.window.sentences_view.row_widgets[0]
        edit = next(button for button in
                    row.findChildren(fg.QtWidgets.QPushButton)
                    if button.text() == "Edit")
        edit.click()
        self.app.processEvents()

        self.assertEqual(self.window._active_sentence_index, 0)
        self.assertEqual(len(self.window.waveform.segments), 3)
        self.assertEqual(self.window.timing.factors(), [1.0, 1.15, 1.0])
        self.assertTrue(self.window.pitch_track.targets())
        self.assertTrue(self.window.voicing_track.targets())
        self.assertTrue(self.window.vocal_tract_track.targets())

    def test_sentence_and_phrase_batch_removal_are_undoable(self):
        self.window.sentences = [
            self.window._new_sentence_state(text)
            for text in ("one", "two", "three")]
        self.window._active_sentence_index = 0
        self.window._refresh_sentence_selector(0)
        self.window._restore_sentence(0)
        self.window._refresh_sentences_view()
        with mock.patch.object(
                fg.QtWidgets.QMessageBox, "question",
                return_value=fg.QtWidgets.QMessageBox.Yes):
            self.assertTrue(
                self.window._remove_selected_sentences([0, 2]))
        self.assertEqual([state["text"] for state in self.window.sentences],
                         ["two"])
        self.window.undo_stack.undo()
        self.assertEqual([state["text"] for state in self.window.sentences],
                         ["one", "two", "three"])

        state = self.window.sentences[0]
        state["phrases"] = [
            self.window._new_phrase_state("first"),
            self.window._new_phrase_state("second"),
            self.window._new_phrase_state("third")]
        state["text"] = (
            "first [pau] [pau] second [pau] [pau] third")
        with mock.patch.object(
                fg.QtWidgets.QMessageBox, "question",
                return_value=fg.QtWidgets.QMessageBox.Yes):
            self.assertTrue(self.window._remove_selected_phrases(0, [0, 2]))
        self.assertEqual(self.window.sentences[0]["text"], "second")
        self.window.undo_stack.undo()
        self.assertEqual(
            [phrase["text"] for phrase in self.window.sentences[0]["phrases"]],
            ["first", "second", "third"])

    def test_selected_phrase_chips_drag_as_one_group(self):
        phrases = [{"id": value, "text": value}
                   for value in ("a", "b", "c", "d")]
        board = fg.PhraseBoard(phrases)
        board.resize(800, 90)
        board.show()
        self.app.processEvents()
        board._selected = {0, 2}
        board._refresh_selection()
        orders = []
        board.orderChanged.connect(orders.append)
        mime = fg.QtCore.QMimeData()
        mime.setData("application/x-festvox-phrase", b"a")
        event = fg.QtGui.QDropEvent(
            fg.QtCore.QPointF(799, 45), fg.Qt.MoveAction, mime,
            fg.Qt.LeftButton, fg.Qt.NoModifier)

        board.dropEvent(event)

        self.assertEqual(orders, [["b", "d", "a", "c"]])
        board.deleteLater()

    def test_phrase_board_wraps_and_sentence_has_one_speaker_badge(self):
        state = self.window._new_sentence_state("many pauses")
        state["phrases"] = [
            self.window._new_phrase_state("phrase %d" % index)
            for index in range(8)]
        row = fg.SentenceRow(0, state)
        row.resize(650, 500)
        row.show()
        self.app.processEvents()

        tops = {chip.geometry().top() for chip in row.board._chips}
        badges = row.findChildren(fg.SpeakerBadge)

        self.assertGreater(len(tops), 1)
        self.assertGreater(row.board.minimumHeight(),
                           row.board._chips[0].sizeHint().height())
        self.assertEqual(len(badges), 1)
        row.deleteLater()

    def test_waveform_click_selects_phone_and_playback_completion_rewinds(self):
        segments = [fc.Segment("pau", 0, .1),
                    fc.Segment("a", .1, .3),
                    fc.Segment("pau", .3, .4)]
        self.window.waveform.set_synthesis(fc.Synthesis(
            np.zeros(400, np.float32), 1000, segments))

        self.window.waveform._on_selection_drag(.2, .2, True)
        self.assertEqual(self.window.waveform.selected_range, (1, 1))
        self.window.waveform.set_playhead(.35)
        self.window._playback_timeline_start = .35
        token = self.window._playback_token
        self.window._finish_playback(token)
        self.assertEqual(self.window.waveform.playhead_time(), 0.0)

    def test_single_phone_selection_is_cosmetic_for_timing_drag(self):
        for phone in ("t", "a"):
            with self.subTest(phone=phone):
                segments = [fc.Segment("pau", 0, .1),
                            fc.Segment(phone, .1, .2),
                            fc.Segment("i", .2, .4),
                            fc.Segment("pau", .4, .5)]
                waveform = self.window.waveform
                waveform.set_synthesis(fc.Synthesis(
                    np.zeros(500, np.float32), 1000, segments))
                waveform.selected_range = (1, 1)
                waveform._highlight_selection()
                boundary = waveform.boundaries[1]
                boundary.setValue(.28)

                with mock.patch.object(
                        fg.QtWidgets.QApplication, "keyboardModifiers",
                        return_value=fg.Qt.NoModifier):
                    waveform._on_drag_finish(boundary)

                self.assertEqual(waveform.selected_range, (1, 1))
                self.assertFalse(waveform._selection_spans_multiple_phones())
                self.assertIsNone(waveform._boundary_drag_snapshot)
                self.assertAlmostEqual(waveform.segments[1].end, .28)
                self.assertAlmostEqual(waveform.segments[2].start, .28)
                self.assertAlmostEqual(waveform.segments[2].end, .48)

    def test_space_stops_playback_and_holds_current_playhead(self):
        self.window.waveform.sr = 1000
        self.window.waveform.audio = np.zeros(1000, np.float32)
        self.window.waveform.set_workspace_duration(1.0)
        self.window.waveform.set_playhead(.25)
        self.window._playback_timeline_start = .25
        self.window._set_playback_active(True)
        self.window.text.setFocus()
        event = fg.QtGui.QKeyEvent(
            fg.QtCore.QEvent.KeyPress, fg.Qt.Key_Space,
            fg.Qt.NoModifier)

        with mock.patch.object(
                self.window, "_advance_playhead",
                side_effect=lambda: self.window.waveform.set_playhead(.47)) \
                as advance, mock.patch.object(
                    self.window.player, "stop") as stop:
            handled = self.window.eventFilter(self.window.text, event)

        self.assertTrue(handled)
        advance.assert_called_once_with()
        stop.assert_called_once_with()
        self.assertAlmostEqual(self.window.waveform.playhead_time(), .47)
        self.assertFalse(self.window._playback_active)
        self.assertEqual(self.window.statusBar().currentMessage(),
                         "Status: stopped at 0.47s")

    def test_pitch_reset_rerenders_from_generated_reference(self):
        segments = [fc.Segment("pau", 0.0, 0.1),
                    fc.Segment("a", 0.1, 0.3),
                    fc.Segment("pau", 0.3, 0.4)]
        generated = [(0.1, 170.0), (0.2, 175.0), (0.3, 165.0)]
        overridden = [(0.1, 240.0), (0.2, 250.0), (0.3, 230.0)]
        syn = fc.Synthesis(
            np.zeros(6400, np.float32), 16000, segments,
            text="a", voicebank="test", targets=overridden,
            generated_targets=generated, pitch_override=overridden,
            pitch_mode="curve")
        self.window._show_synthesis(syn)
        self.window.sentences[0]["rendered"] = True
        self.window._on_pitch_clear()
        captured = {}

        class FakeBackend:
            def synth_phones(_self, phones, voicebank, speed, **kwargs):
                captured.update(kwargs)
                return fc.Synthesis(
                    np.zeros(6400, np.float32), 16000, segments,
                    text="a", voicebank="test", targets=generated,
                    generated_targets=generated)

        self.window._need_backend = lambda: True
        self.window._ab = lambda: FakeBackend()
        self.window._current_voicebank = lambda: "test"
        self.window.on_rerender()

        self.assertEqual(captured["prev_targets"], generated)

    def test_rerender_carries_pitchmark_recovered_english_contour(self):
        segments = [fc.Segment("pau", 0.0, 0.08),
                    fc.Segment("eh", 0.08, 0.28),
                    fc.Segment("pau", 0.28, 0.36)]
        marks = [0.085, 0.091, 0.097, 0.104, 0.111, 0.119, 0.127,
                 0.136, 0.145, 0.155, 0.165, 0.176, 0.187, 0.199,
                 0.211, 0.224, 0.237, 0.251, 0.265]
        synthesis = fc.Synthesis(
            np.zeros(360, np.float32), 1000, segments,
            text="e", lang="en", voicebank="test",
            target_pitchmarks=marks)
        self.window.text.setText("e")
        self.window._show_synthesis(synthesis)
        self.window.sentences[0]["rendered"] = True
        captured = {}

        class FakeBackend:
            def synth_phones(_self, phones, voicebank, speed, **kwargs):
                captured.update(kwargs)
                return fc.Synthesis(
                    np.zeros(360, np.float32), 1000, segments,
                    text="e", lang="en", voicebank="test",
                    targets=list(kwargs.get("prev_targets") or []),
                    generated_targets=list(kwargs.get("prev_targets") or []))

        self.window._need_backend = lambda: True
        self.window._ab = lambda: FakeBackend()
        self.window._current_voicebank = lambda: "test"

        self.window.on_rerender()

        self.assertGreater(len(captured["prev_targets"]), 3)
        self.assertTrue(all(
            fc.PITCH_MIN_HZ <= value <= fc.PITCH_MAX_HZ
            for _time, value in captured["prev_targets"]))

    def test_rerender_retimes_ground_truth_across_double_pause(self):
        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        segments = [fc.Segment("pau", 0.0, 0.04),
                    fc.Segment("pau", 0.04, 0.12),
                    fc.Segment("y", 0.12, 0.20),
                    fc.Segment("uw", 0.20, 0.40),
                    fc.Segment("pau", 0.40, 0.52),
                    fc.Segment("pau", 0.52, 0.76),
                    fc.Segment("dh", 0.76, 0.84),
                    fc.Segment("ih", 0.84, 1.00),
                    fc.Segment("s", 1.00, 1.10),
                    fc.Segment("pau", 1.10, 1.18),
                    fc.Segment("pau", 1.18, 1.25)]
        generated = [(0.12, 190.0), (0.30, 175.0), (0.40, 160.0),
                     (0.76, 198.0), (0.87, 182.0), (1.05, 153.0),
                     (1.10, 145.0)]
        self.window.text.setText("you have ended it. this is the end.")
        self.window._show_synthesis(fc.Synthesis(
            np.zeros(1250, np.float32), 1000, segments,
            text=self.window.text.text(), voicebank="voice",
            targets=generated, generated_targets=generated))
        self.window.waveform.set_factor(3, 2.0)
        seg_durs = self.window._rerender_seg_durs()
        pitch, _fall = self.window._pitch()
        expected = fc.remap_targets_aligned(
            generated, segments, self.window.waveform.segments)
        expected = fc.remap_targets_aligned(
            expected, self.window.waveform.segments,
            fc.segments_from_durations(seg_durs))
        expected = fc.anchor_phrase_targets(seg_durs, expected, pitch)
        captured = {}

        class FakeBackend:
            def synth_phones(_self, phones, voicebank, speed=1.0, **kwargs):
                captured.update(kwargs)
                rendered_segments = fc.segments_from_durations(
                    kwargs["seg_durs"])
                ground = list(kwargs.get("ground_truth_targets") or [])
                sample_count = max(1, int(round(
                    rendered_segments[-1].end * 1000)))
                return fc.Synthesis(
                    np.zeros(sample_count, np.float32), 1000,
                    rendered_segments, text=kwargs.get("text", ""),
                    voicebank=voicebank, phones=list(phones),
                    targets=ground, generated_targets=ground)

        self.window._need_backend = lambda: True
        self.window._ab = lambda: FakeBackend()
        self.window._current_lang_code = lambda: "en"
        self.window._current_voicebank = lambda: "voice"

        self.window.on_rerender()

        np.testing.assert_allclose(
            captured["ground_truth_targets"], expected, rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            captured["prev_targets"], expected, rtol=0, atol=1e-7)
        self.assertTrue(captured["preserve_pitch_register"])
        self.assertEqual(
            [segment.phone for segment in captured["old_segments"]],
            [phone for phone, _duration in seg_durs],
        )
        np.testing.assert_allclose(
            [segment.dur for segment in captured["old_segments"]],
            [duration for _phone, duration in seg_durs],
            rtol=0.0, atol=1.0e-12,
        )
        self.assertEqual([phone for phone, _duration in seg_durs][4:8],
                         ["pau", "pau", "pau", "pau"])
        second = fc.phrase_blocks(
            self.window.current.segments, self.window.text.text())[1]
        second_values = [value for time, value in
                         self.window.current.generated_targets
                         if second["start"] <= time <= second["end"]]
        self.assertGreaterEqual(len(second_values), 3)
        self.assertGreater(max(second_values) - min(second_values), 20.0)

    def test_global_pitch_change_recenters_once_without_follow_up_drift(self):
        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        segments = [
            fc.Segment("pau", 0.00, 0.08, uid="lead"),
            fc.Segment("ih", 0.08, 0.24, uid="vowel"),
            fc.Segment("z", 0.24, 0.32, uid="coda"),
            fc.Segment("pau", 0.32, 0.40, uid="tail"),
        ]
        entries = [(segment.phone, segment.dur) for segment in segments]
        baseline = fc.anchor_phrase_targets(
            entries,
            [(0.08, 172.0), (0.16, 181.0), (0.24, 168.0),
             (0.32, 158.0)],
            165.0,
        )
        self.window.text.setText("is")
        synthesis = fc.Synthesis(
            np.zeros(400, np.float32), 1000, segments,
            text="is", lang="en", voicebank="voice",
            targets=baseline, generated_targets=baseline,
        )
        self.window._show_synthesis(synthesis)
        state = self.window.sentences[0]
        state.update({
            "synthesis": synthesis,
            "rendered": True,
            "pitch_hz": 165.0,
            "rendered_pitch_hz": 165.0,
        })

        captured = []

        class FakeBackend:
            def synth_phones(_self, phones, voicebank, speed=1.0, **kwargs):
                captured.append(copy.deepcopy(kwargs))
                rendered_segments = fc.segments_from_durations(
                    kwargs["seg_durs"])
                ground = list(kwargs.get("ground_truth_targets") or [])
                return fc.Synthesis(
                    np.zeros(400, np.float32), 1000, rendered_segments,
                    text=kwargs.get("text", ""), lang="en",
                    voicebank=voicebank, phones=list(phones),
                    targets=ground, generated_targets=ground,
                )

        self.window._need_backend = lambda: True
        self.window._ab = lambda: FakeBackend()
        self.window._current_lang_code = lambda: "en"
        self.window._current_voicebank = lambda: "voice"
        self.window.pitch.blockSignals(True)
        self.window.pitch.setValue(220.0)
        self.window.pitch.blockSignals(False)
        self.window._on_pitch_parameter_changed()

        self.window.on_rerender()
        first = list(captured[-1]["prev_targets"])
        expected = fc.pitch_domain.recenter_targets_log(
            fc.anchor_phrase_targets(entries, baseline, 220.0),
            220.0, fc.PITCH_MIN_HZ, fc.PITCH_MAX_HZ)
        np.testing.assert_allclose(first, expected, rtol=0.0, atol=1.0e-7)
        self.assertAlmostEqual(
            fc.pitch_domain.geometric_mean_hz(
                value for _time, value in first),
            220.0,
        )
        self.assertEqual(state["rendered_pitch_hz"], 220.0)

        self.window.on_rerender()
        second = list(captured[-1]["prev_targets"])
        np.testing.assert_allclose(second, first, rtol=0.0, atol=1.0e-7)
        self.assertTrue(captured[-1]["preserve_pitch_register"])

    def test_left_selection_stretch_keeps_right_edge_anchored(self):
        segments = [fc.Segment("s", 0, .1), fc.Segment("a", .1, .3),
                    fc.Segment("t", .3, .4), fc.Segment("i", .4, .6),
                    fc.Segment("pau", .6, .7)]
        syn = fc.Synthesis(np.zeros(700, np.float32), 1000, segments)
        self.window.waveform.set_synthesis(syn)
        self.window.waveform.selected_range = (1, 3)

        self.window.waveform._stretch_selection(.05, from_left=True)

        edited = self.window.waveform.segments
        self.assertAlmostEqual(edited[3].end, .6)
        self.assertAlmostEqual(edited[0].end, edited[1].start)
        self.assertAlmostEqual(edited[2].dur, .1)
        self.assertGreater(edited[1].dur + edited[3].dur, .4)

    def test_selection_preview_shifts_later_boundaries_only_once(self):
        segments = [fc.Segment("pau", 0, .1),
                    fc.Segment("a", .1, .3),
                    fc.Segment("t", .3, .4),
                    fc.Segment("i", .4, .5)]
        self.window.waveform.set_synthesis(fc.Synthesis(
            np.zeros(500, np.float32), 1000, segments))
        self.window.waveform.selected_range = (1, 2)

        self.window.waveform._stretch_selection(.6)
        self.window.waveform._stretch_selection(.6)

        edited = self.window.waveform.segments
        self.assertAlmostEqual(edited[2].end, .6)
        self.assertAlmostEqual(edited[3].start, .6)
        self.assertAlmostEqual(edited[3].end, .7)
        self.assertAlmostEqual(
            float(self.window.waveform.boundaries[2].value()), .6)

    def test_all_pause_selection_can_resize(self):
        segments = [fc.Segment("a", 0, .1),
                    fc.Segment("pau", .1, .2),
                    fc.Segment("pau", .2, .4),
                    fc.Segment("b", .4, .5)]
        self.window.waveform.set_synthesis(fc.Synthesis(
            np.zeros(500, np.float32), 1000, segments))
        self.window.waveform.selected_range = (1, 2)

        self.window.waveform._stretch_selection(.6)

        edited = self.window.waveform.segments
        self.assertAlmostEqual(edited[1].start, .1)
        self.assertAlmostEqual(edited[2].end, .6)
        self.assertAlmostEqual(edited[3].start, .6)
        self.assertGreater(edited[1].dur, .1)
        self.assertGreater(edited[2].dur, .2)

    def test_phrase_sequence_keeps_or_collapses_boundary_pause(self):
        state = self.window._new_sentence_state("one. two.")
        state["phrases"] = [self.window._new_phrase_state("one."),
                            self.window._new_phrase_state("two.")]

        def synth_phrase(text, *_args, **_kwargs):
            phone = "a" if text.startswith("one") else "b"
            return fc.Synthesis(
                np.zeros(40, np.float32), 100,
                [fc.Segment("pau", 0, .1),
                 fc.Segment(phone, .1, .3),
                 fc.Segment("pau", .3, .4)],
                text=text, voicebank="voice")

        self.window._synthesize_phrase_request = synth_phrase
        normal = self.window._generate_phrase_sequence(
            state, "en", "voice", 1.0, 180, 10, False, {}, None)
        faulted = self.window._generate_phrase_sequence(
            state, "en", "voice", 1.0, 180, 10, False,
            {"single_pause": True}, None)

        self.assertEqual([segment.phone for segment in normal.segments],
                         ["pau", "a", "pau", "pau", "pau", "pau",
                          "b", "pau"])
        self.assertEqual([segment.phone for segment in faulted.segments],
                         ["pau", "a", "pau", "b", "pau"])

    def test_asaxi_phrase_sequence_localizes_pitch_edits_and_defers_phonation(
            self):
        state = self.window._new_sentence_state("shěso.\nox.")
        state["lang_code"] = "asaxi"
        state["asaxi_state"] = fg.asaxi_editing.new_edit_state(
            state["text"])
        state["asaxi_state"]["mora_pitch_offsets_cents"] = {
            "0": 90.0,
            "2": -60.0,
        }
        state["asaxi_state"]["mora_tone_overrides"] = {
            "1": "L",
            "2": "H",
        }
        state["phrases"] = [
            self.window._new_phrase_state("shěso."),
            self.window._new_phrase_state("ox."),
        ]
        received_offsets = []
        received_tones = []
        phonation_calls = []

        def synth_phrase(text, *_args, **kwargs):
            received_offsets.append(
                dict(kwargs.get("asaxi_pitch_offsets_cents") or {}))
            received_tones.append(
                dict(kwargs.get("asaxi_tone_overrides") or {}))
            phone = "a" if text.startswith("sh") else "o"
            synthesis = fc.Synthesis(
                np.zeros(30, np.float32), 100,
                [fc.Segment("pau", 0, .1),
                 fc.Segment(phone, .1, .2),
                 fc.Segment("pau", .2, .3)],
                text=text, voicebank="voice", lang="asaxi")
            synthesis.asaxi_prosody = {
                "mora_count": 2 if phone == "a" else 1,
                "moras": [],
            }
            return synthesis

        self.window._synthesize_phrase_request = synth_phrase
        self.window._apply_shared_voicing_stage = (
            lambda synthesis: phonation_calls.append(synthesis) or synthesis
        )

        self.window._generate_phrase_sequence(
            state, "asaxi", "voice", 1.0, 180, 10, False, {}, None)

        self.assertEqual(received_offsets[0][0], 90.0)
        self.assertEqual(received_offsets[1], {0: -60.0})
        self.assertEqual(received_tones[0][1], "L")
        self.assertEqual(received_tones[1], {0: "H"})
        self.assertEqual(phonation_calls, [])

    def test_phrase_fault_false_inherits_sentence_fault(self):
        state = self.window._new_sentence_state("one. two.")
        state["fault_mode"]["legacy_joins"] = True
        state["phrases"] = [self.window._new_phrase_state("one."),
                            self.window._new_phrase_state("two.")]
        # Old projects may contain explicit false values.  They must not mask
        # a later sentence-wide Legacy joins activation.
        state["phrases"][0]["fault_mode"]["legacy_joins"] = False
        seen = []

        def synth_phrase(text, *args, **_kwargs):
            seen.append((text, dict(args[7])))
            return fc.Synthesis(
                np.zeros(20, np.float32), 100,
                [fc.Segment("pau", 0, .1),
                 fc.Segment("a", .1, .2)],
                text=text, voicebank="voice")

        self.window._synthesize_phrase_request = synth_phrase
        self.window._generate_phrase_sequence(
            state, "en", "voice", 1.0, 180, 10, False,
            {"legacy_joins": True}, None)

        self.assertEqual(len(seen), 2)
        self.assertTrue(all(faults["legacy_joins"]
                            for _text, faults in seen))

    def test_unchecked_phrase_fault_removes_local_override(self):
        state = self.window._new_sentence_state("one.")
        state["phrases"] = [self.window._new_phrase_state("one.")]
        state["phrases"][0]["fault_mode"]["legacy_joins"] = True
        self.window.sentences = [state]

        self.window._set_phrase_fault(0, 0, "legacy_joins", False)

        self.assertNotIn(
            "legacy_joins", state["phrases"][0]["fault_mode"])

    def test_clear_all_faults_also_clears_phrase_faults(self):
        state = self.window._new_sentence_state("one.")
        state["phrases"] = [self.window._new_phrase_state("one.")]
        state["phrases"][0]["fault_mode"]["legacy_joins"] = True
        self.window.sentences = [state]

        self.window._clear_faults_from_all_sentences()

        self.assertEqual(state["phrases"][0]["fault_mode"], {})

    def test_cache_only_project_row_restores_audio(self):
        with tempfile.TemporaryDirectory() as root:
            wav = Path(root) / "cached.wav"
            fc.write_wav(wav, np.ones(100, np.float32) * .1, 1000)
            row = {"text": "reordered phrases", "segments": [],
                   "phones": [], "cache_wav": "cached.wav",
                   "needs_rerender": True}

            state = self.window._state_from_project_row(row, root)

            self.assertTrue(state["cache_loaded"])
            self.assertTrue(state["rendered"])
            self.assertTrue(state["needs_rerender"])
            self.assertEqual(len(state["synthesis"].samples), 100)
            self.assertEqual(state["synthesis"].segments, [])

    def test_duplicate_survives_rerender_with_occurrence_ids(self):
        segments = [fc.Segment("pau", 0, .1),
                    fc.Segment("a", .1, .2),
                    fc.Segment("b", .2, .3),
                    fc.Segment("pau", .3, .4)]
        syn = fc.Synthesis(
            np.linspace(-.2, .2, 400, dtype=np.float32), 1000,
            segments, text="a b", voicebank="voice", phones=["a", "b"])
        self.window._show_synthesis(syn)
        self.window._commit_rendered_state(syn)
        self.window._capture_active_sentence()
        original_ids = [segment.uid
                        for segment in self.window.waveform.segments]
        self.window.waveform._set_selected_range(2, 2)
        self.assertTrue(self.window._shortcut_duplicate())
        expected_phones = [segment.phone
                           for segment in self.window.waveform.segments]
        expected_ids = [segment.uid
                        for segment in self.window.waveform.segments]

        class FakeBackend:
            def synth_phones(_self, phones, voicebank, speed=1.0, **kwargs):
                rendered = fc.segments_from_durations(kwargs["seg_durs"])
                count = max(1, int(round(rendered[-1].end * 1000)))
                return fc.Synthesis(
                    np.ones(count, np.float32) * .15, 1000, rendered,
                    text=kwargs.get("text", ""), voicebank=voicebank,
                    phones=list(phones))

        self.window._need_backend = lambda: True
        self.window._ab = lambda: FakeBackend()
        self.window._current_voicebank = lambda: "voice"
        self.window._refresh_voice_metadata = lambda: None

        self.window.on_rerender()

        self.assertEqual([segment.phone for segment in
                          self.window.waveform.segments], expected_phones)
        self.assertEqual([segment.uid for segment in
                          self.window.waveform.segments], expected_ids)
        self.assertEqual([segment.uid for segment in
                          self.window.sentences[0]["editor_segments"]],
                         expected_ids)
        self.window.undo_stack.undo()
        self.assertEqual([segment.uid for segment in
                          self.window.waveform.segments], original_ids)
        self.window.undo_stack.redo()
        self.assertEqual([segment.uid for segment in
                          self.window.waveform.segments], expected_ids)

    def test_duplicated_sentence_gets_fresh_segment_ids(self):
        rendered = [fc.Segment("pau", 0, .1),
                    fc.Segment("a", .1, .2),
                    fc.Segment("pau", .2, .3)]
        syn = fc.Synthesis(np.zeros(300, np.float32), 1000, rendered)
        state = self.window._new_sentence_state("a")
        state["synthesis"] = syn
        state["editor_segments"] = copy.deepcopy(rendered)

        duplicate = self.window._fresh_sentence_copy(state)

        source_ids = [segment.uid for segment in state["editor_segments"]]
        editor_ids = [segment.uid for segment in
                      duplicate["editor_segments"]]
        rendered_ids = [segment.uid for segment in
                        duplicate["synthesis"].segments]
        self.assertEqual(editor_ids, rendered_ids)
        self.assertTrue(set(source_ids).isdisjoint(editor_ids))

    def test_folder_project_round_trip_keeps_duplicated_editor_regions(self):
        segments = [fc.Segment("pau", 0, .1),
                    fc.Segment("a", .1, .2),
                    fc.Segment("b", .2, .3),
                    fc.Segment("pau", .3, .4)]
        syn = fc.Synthesis(
            np.linspace(-.25, .25, 400, dtype=np.float32), 1000,
            segments, text="a b", voicebank="voice", phones=["a", "b"],
            target_pitchmarks=[.105, .115, .205, .215],
             splice_records=[{
                 "segment_index": 1, "time": .15,
                 "handoff_start": .145, "handoff_end": .155,
                 "position_source": "festival-us-map", "estimated": False,
             }],
             frame_trajectory_records=[{
                 "target_index": 2, "time": .205,
                 "previous_source_frame": 12, "source_frame": 16,
                 "centre_offset_samples": -4,
                 "original_correlation": .31,
                 "corrected_correlation": .94,
                 "correlation_improvement": .63,
                 "phone": "b",
                 "reason": "phase-reference-corrected",
             }],
             vowel_realizations=[{
                 "segment_index": 1, "mora_index": 0, "phone": "i",
                 "strategy": "source_filter_residual_devoiced",
                 "reason": "test decision", "periodicity_before": .81,
                 "periodicity_after": .22,
             }])
        self.window.text.setText("a b")
        self.window.sentences[0]["text"] = "a b"
        self.window._show_synthesis(syn)
        self.window._commit_rendered_state(syn)
        self.window._capture_active_sentence()
        self.window.waveform._set_selected_range(1, 1)
        self.assertTrue(self.window._shortcut_duplicate())
        expected_phones = [segment.phone
                           for segment in self.window.waveform.segments]
        expected_ids = [segment.uid
                        for segment in self.window.waveform.segments]
        expected_audio = self.window.waveform.audio.copy()

        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "Duplicated Regions"
            self.assertTrue(self.window.on_save_project(project))
            self.assertTrue((project / "project.json").is_file())
            self.assertTrue((project / "cache" / "sentence_0001.wav").is_file())
            self.assertTrue((project / "exports").is_dir())
            self.window.waveform.delete_selection()

            self.assertTrue(
                self.window.on_open_project(project / "project.json"))

        self.assertEqual([segment.phone for segment in
                          self.window.waveform.segments], expected_phones)
        self.assertEqual(self.window.sentences[0]["rendered_text"], "a b")
        self.assertEqual(self.window.current.target_pitchmarks,
                         [.105, .115, .205, .215])
        self.assertEqual(
            self.window.current.splice_records[0]["position_source"],
            "festival-us-map")
        self.assertEqual(
            self.window.current.frame_trajectory_records[0][
                "centre_offset_samples"], -4)
        self.assertEqual(
            self.window.current.vowel_realizations[0]["strategy"],
            "source_filter_residual_devoiced")
        self.assertEqual([segment.uid for segment in
                          self.window.waveform.segments], expected_ids)
        np.testing.assert_allclose(
            self.window.sentences[0]["preview_audio"], expected_audio,
            rtol=0, atol=4e-5)

    def test_project_resave_removes_only_stale_sentence_cache_wavs(self):
        syn = fc.Synthesis(
            np.linspace(-.1, .1, 200, dtype=np.float32), 1000,
            [fc.Segment("a", 0, .2)], text="a", phones=["a"])
        self.window.sentences[0]["text"] = "a"
        self.window._show_synthesis(syn)
        self.window._commit_rendered_state(syn)

        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "Cache Cleanup"
            self.assertTrue(self.window.on_save_project(project))
            stale = project / "cache" / "sentence_9999.wav"
            keep = project / "cache" / "analysis.wav"
            stale.write_bytes(b"stale")
            keep.write_bytes(b"intentional")

            self.assertTrue(self.window.on_save_project(project))

            self.assertFalse(stale.exists())
            self.assertEqual(keep.read_bytes(), b"intentional")

    def test_open_project_json_rejects_pre_version_four_files(self):
        with tempfile.TemporaryDirectory() as root:
            legacy = Path(root) / "old.json"
            fc.save_batch_project(legacy, [{"text": "old"}])
            with mock.patch.object(
                    fg.QtWidgets.QMessageBox, "warning") as warning:
                opened = self.window.on_open_project(legacy)

        self.assertFalse(opened)
        self.assertIn("Only version-4", warning.call_args.args[2])

    def test_sentences_generate_warns_before_resetting_manual_edits(self):
        syn = fc.Synthesis(
            np.ones(300, np.float32) * .1, 1000,
            [fc.Segment("pau", 0, .1), fc.Segment("a", .1, .2),
             fc.Segment("pau", .2, .3)],
            text="a", voicebank="voice", phones=["a"])
        self.window._show_synthesis(syn)
        self.window._commit_rendered_state(syn)
        state = self.window.sentences[0]
        self.window._set_state_pending(state, "generate", "Text changed")
        self.window._need_backend = lambda: True

        with mock.patch.object(
                fg.QtWidgets.QMessageBox, "question",
                return_value=fg.QtWidgets.QMessageBox.No) as question:
            self.window._generate_sentence(0)

        self.assertEqual(
            question.call_args.args[2],
            "Generate may reset manual timing, pitch, segment, or recording "
            "edits. Continue?")
        self.assertIs(self.window.sentences[0]["synthesis"], syn)

    def test_generate_commits_latest_audio_for_both_tabs_and_export(self):
        old = fc.Synthesis(
            np.ones(200, np.float32) * -.1, 1000,
            [fc.Segment("pau", 0, .05), fc.Segment("a", .05, .15),
             fc.Segment("pau", .15, .2)],
            text="old", voicebank="voice", phones=["a"])
        self.window._show_synthesis(old)
        self.window._commit_rendered_state(old)
        self.window.text.setText("new")
        newest = np.linspace(-.3, .3, 300, dtype=np.float32)

        class FakeBackend:
            def synth(_self, text, lang, voicebank, speed=1.0, **_kwargs):
                return fc.Synthesis(
                    newest.copy(), 1000,
                    [fc.Segment("pau", 0, .05),
                     fc.Segment("n", .05, .25),
                     fc.Segment("pau", .25, .3)],
                    text=text, lang=lang, voicebank=voicebank, phones=["n"])

        self.window._need_backend = lambda: True
        self.window._ab = lambda: FakeBackend()
        self.window._current_voicebank = lambda: "voice"
        self.window._refresh_voice_metadata = lambda: None

        rendered = self.window._generate_current(confirm_replace=False)

        self.assertIsNotNone(rendered)
        state = self.window.sentences[0]
        np.testing.assert_allclose(state["preview_audio"], newest)
        np.testing.assert_allclose(self.window._output_audio()[0], newest)
        state["synthesis"].samples = np.ones(10, np.float32) * .75
        played = {}
        with mock.patch.object(
                self.window, "_start_playback",
                side_effect=lambda samples, sr, **_kwargs:
                played.update(samples=np.asarray(samples).copy(), sr=sr)):
            self.window._play_sentence_indices([0])
        np.testing.assert_allclose(played["samples"], newest)
        self.assertEqual(played["sr"], 1000)

        with tempfile.TemporaryDirectory() as root:
            exported = Path(root) / "latest.wav"
            with mock.patch.object(
                    fg.QtWidgets.QFileDialog, "getSaveFileName",
                    return_value=(str(exported), "WAV (*.wav)")):
                self.window.on_export()
            samples, sr = fc.read_wav(str(exported))
        self.assertEqual(sr, 1000)
        np.testing.assert_allclose(samples, newest, rtol=0, atol=4e-5)

    def test_japanese_parameter_page_is_contextual(self):
        row = self.window.parameter_mode.findData("japanese")
        voicing_row = self.window.parameter_mode.findData("mora_voicing")
        self.assertGreaterEqual(row, 0)
        self.assertGreaterEqual(voicing_row, 0)
        for index in (row, voicing_row):
            self.assertFalse(
                self.window.parameter_mode.model().item(index).isEnabled())

        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        japanese = self.window.lang.findText("Japanese")
        self.assertGreaterEqual(japanese, 0)
        self.window.lang.blockSignals(True)
        self.window.lang.setCurrentIndex(japanese)
        self.window.lang.blockSignals(False)
        compatibility = mock.Mock()
        compatibility.supports.side_effect = lambda language: language == "ja"
        preserved_state = {"0": 0.4}
        self.window.sentences[0]["japanese_state"][
            "mora_voicing_overrides"] = preserved_state.copy()
        with mock.patch.object(
                self.window.fest, "voice_compatibility",
                return_value=compatibility):
            self.window._update_parameter_availability()

            for index in (row, voicing_row):
                self.assertTrue(
                    self.window.parameter_mode.model().item(index).isEnabled())
        for index, mode in ((row, "accent"),
                            (voicing_row, "mora_voicing")):
            self.window.parameter_mode.setCurrentIndex(index)
            self.assertIs(self.window.parameter_stack.currentWidget(),
                          self.window.japanese_page)
            self.assertEqual(self.window.japanese_editor._edit_mode, mode)
        english = self.window.lang.findText("English")
        self.assertGreaterEqual(english, 0)
        self.window.lang.blockSignals(True)
        self.window.lang.setCurrentIndex(english)
        self.window.lang.blockSignals(False)
        with mock.patch.object(
                self.window.fest, "voice_compatibility",
                return_value=compatibility):
            self.window._update_parameter_availability()
        for index in (row, voicing_row):
            self.assertFalse(
                self.window.parameter_mode.model().item(index).isEnabled())
        self.assertNotIn(self.window.parameter_mode.currentData(),
                         {"japanese", "mora_voicing"})
        self.assertEqual(
            self.window.sentences[0]["japanese_state"]
            ["mora_voicing_overrides"], preserved_state)

    def test_shared_mora_parameters_route_asaxi_and_preserve_state(self):
        pitch_row = self.window.parameter_mode.findData("japanese")
        voicing_row = self.window.parameter_mode.findData("mora_voicing")
        self.assertEqual(self.window.parameter_mode.findData("asaxi_mora"), -1)
        for row in (pitch_row, voicing_row):
            self.assertGreaterEqual(row, 0)
            self.assertFalse(
                self.window.parameter_mode.model().item(row).isEnabled())
        self.window.engine.blockSignals(True)
        self.window.engine.setCurrentIndex(
            self.window.engine.findData("festival_wsl"))
        self.window.engine.blockSignals(False)
        asaxi = self.window.lang.findText("Asaxi")
        self.assertGreaterEqual(asaxi, 0)
        self.window.lang.blockSignals(True)
        self.window.lang.setCurrentIndex(asaxi)
        self.window.lang.blockSignals(False)
        compatibility = mock.Mock()
        compatibility.supports.side_effect = \
            lambda language: language == "asaxi"
        self.window.sentences[0]["asaxi_state"][
            "mora_tone_overrides"] = {"0": "L"}
        with mock.patch.object(
                self.window.fest, "voice_compatibility",
                return_value=compatibility):
            self.window._update_parameter_availability()
        for row in (pitch_row, voicing_row):
            self.assertTrue(
                self.window.parameter_mode.model().item(row).isEnabled())

        self.window.parameter_mode.setCurrentIndex(pitch_row)
        self.assertIs(self.window.parameter_stack.currentWidget(),
                      self.window.asaxi_page)
        self.assertEqual(self.window.asaxi_editor._edit_mode, "accent")
        self.assertEqual(self.window.asaxi_editor.grid.edit_mode, "accent")
        self.assertFalse(self.window.asaxi_editor.tone.isHidden())
        self.assertFalse(self.window.asaxi_editor.mora_pitch.isHidden())
        self.assertTrue(self.window.asaxi_editor.voicing.isHidden())
        self.assertFalse(hasattr(self.window.asaxi_editor, "breathiness"))

        self.window.parameter_mode.setCurrentIndex(voicing_row)
        self.assertIs(self.window.parameter_stack.currentWidget(),
                      self.window.asaxi_page)
        self.assertEqual(self.window.asaxi_editor._edit_mode, "mora_voicing")
        self.assertEqual(self.window.asaxi_editor.grid.edit_mode, "voicing")
        self.assertTrue(self.window.asaxi_editor.tone.isHidden())
        self.assertTrue(self.window.asaxi_editor.mora_pitch.isHidden())
        self.assertFalse(self.window.asaxi_editor.voicing.isHidden())

        english = self.window.lang.findText("English")
        self.window.lang.blockSignals(True)
        self.window.lang.setCurrentIndex(english)
        self.window.lang.blockSignals(False)
        with mock.patch.object(
                self.window.fest, "voice_compatibility",
                return_value=compatibility):
            self.window._update_parameter_availability()

        for row in (pitch_row, voicing_row):
            self.assertFalse(
                self.window.parameter_mode.model().item(row).isEnabled())
        self.assertNotIn(self.window.parameter_mode.currentData(),
                         {"japanese", "mora_voicing"})
        self.assertEqual(
            self.window.sentences[0]["asaxi_state"]
            ["mora_tone_overrides"],
            {"0": "L"},
        )

    def test_asaxi_rerender_rebuilds_pitch_on_edited_timing(self):
        dictionary = fg.asaxi_prosody.load_dictionary()
        plan = fg.asaxi_prosody.analyze_utterance("shěso", dictionary)
        entries = [("pau", 0.08)]
        entries.extend((phone, 0.045 + index * 0.007)
                       for index, phone in enumerate(plan.phones))
        entries.append(("pau", 0.11))
        segments = fc.segments_from_durations(entries)
        original_timing = [
            (segment.phone, segment.start, segment.end)
            for segment in segments
        ]
        state = self.window._new_sentence_state("shěso")
        state["lang_code"] = "asaxi"
        state["asaxi_state"] = fg.asaxi_editing.new_edit_state("shěso")

        baseline, _metadata = self.window._prepare_asaxi_rerender(
            state, "shěso", segments, 165.0, 18.0)
        state["asaxi_state"]["mora_pitch_offsets_cents"] = {"0": 1200.0}
        edited, metadata = self.window._prepare_asaxi_rerender(
            state, "shěso", segments, 165.0, 18.0)

        first = metadata["moras"][0]
        midpoint = (float(first["start"]) + float(first["end"])) / 2.0
        baseline_hz = min(baseline, key=lambda row: abs(row[0] - midpoint))[1]
        edited_hz = min(edited, key=lambda row: abs(row[0] - midpoint))[1]
        self.assertAlmostEqual(edited_hz / baseline_hz, 2.0, places=5)
        self.assertEqual(
            metadata["pitch_model_id"],
            "asaxi-hierarchical-log-f0-v1",
        )
        self.assertEqual(
            metadata["prosody_trace"]["cumulative_frequency_drift"],
            "disabled",
        )
        self.assertTrue(metadata["prosody_trace"]["trajectory"])
        self.assertEqual(
            [(segment.phone, segment.start, segment.end)
             for segment in segments],
            original_timing,
        )

    def test_asaxi_mora_edit_is_undoable_and_marks_rerender(self):
        metadata = {
            "rendered_phones": ["sh", "er", "s"],
            "moras": [{
                "mora_index": 0,
                "phrase_index": 0,
                "word": "shěso",
                "text": "shě",
                "phones": ["sh", "er"],
                "pitch": "H",
                "accentable": True,
                "segment_indices": [0, 1],
            }],
            "mora_phonation_predictions": [{
                "mora_index": 0,
                "eligible": True,
                "automatic_voicing": 0.18,
                "automatic_effective_voicing": 0.18,
                "automatic_breathiness": 0.0,
                "reasons": ["vowel between voiceless sh and s"],
            }],
        }
        state = self.window.sentences[0]
        state["text"] = "shěso"
        state["rendered"] = True
        state["synthesis"] = fc.Synthesis(
            np.zeros(1800, np.float32),
            10000,
            [
                fc.Segment("sh", 0.0, 0.05),
                fc.Segment("er", 0.05, 0.13),
                fc.Segment("s", 0.13, 0.18),
            ],
            text="shěso",
            lang="asaxi",
        )
        state["asaxi_state"] = fg.asaxi_editing.reconcile_plan(
            state["asaxi_state"], "shěso", metadata)
        self.window.asaxi_editor.set_state(state["asaxi_state"])

        self.window._on_asaxi_mora_edit("tone", [0], "L")

        self.assertEqual(
            state["asaxi_state"]["mora_tone_overrides"], {"0": "L"})
        self.assertTrue(state["needs_rerender"])
        self.window.undo_stack.undo()
        self.assertEqual(
            state["asaxi_state"]["mora_tone_overrides"], {})
        self.window.undo_stack.redo()
        self.assertEqual(
            state["asaxi_state"]["mora_tone_overrides"], {"0": "L"})

    def test_asaxi_mora_voicing_edit_updates_generated_curve_immediately(self):
        metadata = {
            "moras": [{
                "mora_index": 0,
                "phrase_index": 0,
                "word": "shěso",
                "text": "shě",
                "phones": ["sh", "er"],
                "pitch": "H",
                "accentable": True,
                "segment_indices": [0, 1],
            }],
            "mora_phonation_predictions": [{
                "mora_index": 0,
                "eligible": True,
                "automatic_voicing": 0.18,
                "automatic_effective_voicing": 0.18,
                "automatic_breathiness": 0.0,
                "reasons": ["vowel between voiceless sh and s"],
            }],
        }
        state = self.window.sentences[0]
        state["text"] = "shěso"
        state["rendered"] = True
        syn = fc.Synthesis(
            np.zeros(1800, np.float32),
            10000,
            [
                fc.Segment("sh", 0.0, 0.05),
                fc.Segment("er", 0.05, 0.13),
                fc.Segment("s", 0.13, 0.18),
            ],
            text="shěso",
            lang="asaxi",
        )
        syn.source_voicing_targets = [
            (index / 1000.0, 0.95) for index in range(181)
        ]
        syn.generated_voicing_targets = list(syn.source_voicing_targets)
        syn.asaxi_prosody = copy.deepcopy(metadata)
        state["synthesis"] = syn
        state["asaxi_state"] = fg.asaxi_editing.reconcile_plan(
            state["asaxi_state"], "shěso", metadata)

        self.window._on_asaxi_mora_edit("voicing", [0], 0.6)

        self.assertAlmostEqual(
            min(value for _time, value in syn.generated_voicing_targets),
            0.6,
            places=2,
        )
        self.assertEqual(
            state["asaxi_state"]["mora_voicing_overrides"], {"0": 0.6})
        self.assertTrue(state["needs_rerender"])
        self.window.undo_stack.undo()
        self.assertLess(
            min(value for _time, value in syn.generated_voicing_targets),
            0.3,
        )

    def test_generated_voice_manifest_constrains_and_selects_language(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "voice"
            (root / "dic").mkdir(parents=True)
            runtime = {
                "source_bundle_id": "srb_test",
                "configuration_id": "vcfg_test",
                "primary_language": "ja",
                "supported_languages": ["ja"],
                "alias_system": "utau-japanese-vcv-v1",
                "voice_entry_points": {"ja": "voice_fixture_ja"},
                "phones": ["pau", "a", "k"],
            }
            metadata_path = root / "dic" / "diphone_index.json"
            metadata_path.write_text(json.dumps(runtime), encoding="utf-8")
            self.window.cfg["festival_wsl"]["voices"] = {
                "fixture": {
                    "dir": str(root), "voice": "voice_fixture_ja",
                    "scm": "festvox/fixture_ja.scm",
                }
            }
            self.window.engine.blockSignals(True)
            self.window.engine.setCurrentIndex(
                self.window.engine.findData("festival_wsl"))
            self.window.engine.blockSignals(False)
            self.window._refresh_voicebanks()

            self.assertEqual(self.window._current_lang_code(), "ja")
            english = self.window.lang.findText("English")
            self.assertFalse(self.window.lang.model().item(english).isEnabled())
            self.assertNotIn(
                "legacy", self.window.voicebank.currentItem().text().casefold()
            )

            metadata_path.write_text(json.dumps({
                "language": "ja",
                "voice_entry_point": "voice_fixture_ja",
            }), encoding="utf-8")
            self.window.fest.invalidate_voice_metadata("fixture")
            self.window._refresh_voicebanks(keep="fixture")
            self.assertIn(
                "legacy metadata",
                self.window.voicebank.currentItem().text().casefold(),
            )

    def test_japanese_edits_are_undoable_and_preserve_manual_candidates(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "きゃく。ねこ？", mode="kana")
        state = self.window.sentences[0]
        state["japanese_state"] = fg.je.new_edit_state(
            utterance, frontend_mode="kana")
        state["japanese_state"]["manual_candidate_overrides"] = {
            "0": "jc_manual"}
        accent = utterance.accent_phrases[0]
        syn = fc.Synthesis(
            np.zeros(300, np.float32), 1000,
            [fc.Segment("pau", 0.0, 0.1),
             fc.Segment("k", 0.1, 0.2),
             fc.Segment("a", 0.2, 0.3)],
            text=utterance.source_text, lang="ja", voicebank="test",
            unit_overrides={1: "k__manual"})
        state["synthesis"] = syn
        state["rendered"] = True
        self.window.current = syn
        self.window._clear_state_pending(state)
        self.window.japanese_editor.set_state(state["japanese_state"])

        self.window._on_japanese_edit(
            "accent", accent.index,
            {"accent_state": "accented", "accent_nucleus": 0})

        edited = state["japanese_state"]
        self.assertEqual(
            edited["accent_overrides"][str(accent.index)]["accent_state"],
            "accented")
        self.assertEqual(edited["manual_candidate_overrides"],
                         {"0": "jc_manual"})
        self.assertEqual(self.window.current.unit_overrides,
                         {1: "k__manual"})
        self.assertEqual(self.window._pending_action(state), "rerender")
        self.window.undo_stack.undo()
        self.assertNotIn(str(accent.index),
                         state["japanese_state"]["accent_overrides"])
        self.assertEqual(state["japanese_state"][
            "manual_candidate_overrides"], {"0": "jc_manual"})
        self.assertEqual(self.window.current.unit_overrides,
                         {1: "k__manual"})
        self.window.undo_stack.redo()
        self.assertIn(str(accent.index),
                      state["japanese_state"]["accent_overrides"])

    def test_multi_mora_voicing_edit_is_undoable_and_sentence_scoped(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "\u304d\u304f", mode="kana")
        state = self.window.sentences[0]
        overlay = fg.je.new_edit_state(utterance, frontend_mode="kana")
        plan = fg.je.create_edited_plan(
            utterance, overlay, runtime_metadata={"language": "ja"})
        plan_state = plan.to_dict()
        plan_state["mora_voicing_predictions"] = [
            item.to_dict() for item in
            fg.japanese_devoicing.predict_mora_voicing(plan)
        ]
        overlay["last_plan"] = plan_state
        state["japanese_state"] = overlay
        segments = fc.segments_from_durations(plan.segment_durations)
        duration = segments[-1].end
        syn = fc.Synthesis(
            np.zeros(max(1, round(duration * 1000)), np.float32),
            1000, segments, text=utterance.source_text, lang="ja")
        state["synthesis"] = syn
        state["rendered"] = True
        self.window.current = syn
        self.window._clear_state_pending(state)
        self.window.japanese_editor.set_state(overlay)
        self.window.japanese_editor.set_edit_mode("mora_voicing")
        indexes = [mora.index for mora in utterance.moras[:2]]
        self.window.japanese_editor.grid.selected_moras = set(indexes)
        self.window.japanese_editor._select_moras(indexes)

        self.window.japanese_editor._mora_voicing_changed(35.0)

        self.assertEqual(
            state["japanese_state"]["mora_voicing_overrides"],
            {str(index): 0.35 for index in indexes},
        )
        self.assertEqual(self.window._pending_action(state), "rerender")
        self.window.undo_stack.undo()
        self.assertEqual(
            state["japanese_state"]["mora_voicing_overrides"], {})
        self.window.undo_stack.redo()
        self.assertEqual(
            state["japanese_state"]["mora_voicing_overrides"],
            {str(index): 0.35 for index in indexes},
        )

    def test_japanese_phrase_structure_and_nucleus_edits_are_undoable(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "\u304b\u306a\u304b\u306a", mode="kana")
        state = self.window.sentences[0]
        state["japanese_state"] = fg.je.new_edit_state(
            utterance, frontend_mode="kana")
        self.window.japanese_editor.set_state(state["japanese_state"])
        phrase = utterance.phrases[0]
        split_mora = phrase.moras[2]

        self.window._on_japanese_edit(
            "accent_structure", phrase.index, [split_mora.index])
        edited = fg.je.apply_linguistic_edits(
            utterance, state["japanese_state"])
        self.assertEqual(len(edited.phrases[0].accent_phrases), 2)
        second = edited.phrases[0].accent_phrases[1]

        self.window._on_japanese_edit(
            "accent", second.index,
            {"accent_state": "accented", "accent_nucleus": 1})
        applied = fg.je.apply_linguistic_edits(
            utterance, state["japanese_state"])
        self.assertEqual(applied.phrases[0].accent_phrases[1]
                         .accent_nucleus, 1)

        self.window.undo_stack.undo()
        self.assertNotIn(
            str(second.index),
            state["japanese_state"]["accent_overrides"])
        self.window.undo_stack.undo()
        self.assertEqual(
            state["japanese_state"]["accent_phrase_boundaries"], {})
        self.window.undo_stack.redo()
        self.assertEqual(
            state["japanese_state"]["accent_phrase_boundaries"]
            [str(phrase.index)], [split_mora.index])

    def test_mora_and_waveform_navigation_share_the_rendered_timeline(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "\u304b\u306a", mode="kana")
        plan = fg.je.create_edited_plan(
            utterance, fg.je.new_edit_state(utterance),
            runtime_metadata={"language": "ja"})
        segments = fc.segments_from_durations(plan.segment_durations)
        duration = segments[-1].end
        syn = fc.Synthesis(
            np.zeros(max(1, int(duration * 1000)), np.float32), 1000,
            segments, text=utterance.source_text, lang="ja")
        state = self.window.sentences[0]
        overlay = fg.je.new_edit_state(utterance, frontend_mode="kana")
        overlay["last_plan"] = plan.to_dict()
        state["japanese_state"] = overlay
        state["synthesis"] = syn
        state["rendered"] = True
        self.window.current = syn
        self.window.waveform.set_synthesis(syn)
        self.window.japanese_editor.set_state(overlay)
        self.window._sync_timing_track(reset=True)

        mora = utterance.moras[1]
        expected = [
            int(row["index"]) for row in plan.to_dict()["segments"]
            if row.get("mora_index") == mora.index
        ]
        self.window._on_japanese_mora_selected(mora.index)
        self.assertEqual(
            self.window.waveform.selected_indices(),
            (expected[0], expected[-1]))
        self.assertIn(mora.index, self.window.japanese_editor.grid._timeline)

        first_mora_edge = next(
            int(row["index"]) for row in plan.to_dict()["segments"]
            if row.get("mora_index") == utterance.moras[0].index)
        self.window.waveform.set_selected(first_mora_edge)
        self.assertEqual(
            self.window.japanese_editor._selected_mora,
            utterance.moras[0].index)
        self.window.waveform.set_playhead(.125)
        self.assertAlmostEqual(
            self.window.japanese_editor.grid._playhead, .125)

    def test_japanese_mora_grid_does_not_repaint_for_hidden_playhead(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "\u304b\u306a" * 120, mode="kana")
        grid = fg.JapaneseMoraGrid()
        try:
            grid.set_model(utterance)
            self.assertEqual(
                len(grid._mora_positions), len(utterance.moras))
            self.assertTrue(grid._accent_ranges)
            with mock.patch.object(grid, "update") as update:
                grid.set_playhead(.125)
                update.assert_not_called()
            self.assertAlmostEqual(grid._playhead, .125)
        finally:
            grid.close()
            grid.deleteLater()
            self.app.processEvents()

    def test_recordings_exposes_every_selected_mora_contribution(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "\u304b", mode="kana")
        state = self.window.sentences[0]
        overlay = fg.je.new_edit_state(utterance, frontend_mode="kana")
        overlay["last_plan"] = {
            "segments": [],
            "source_contributions": {"contributions": [
                {"diphone": "a-k", "mora_indices": [0, 1],
                 "role": "outgoing_vc", "source_alias": "a k"},
                {"diphone": "k-a", "mora_indices": [1, 1],
                 "role": "incoming_cv", "source_alias": "\u304b"},
            ]},
        }
        state["japanese_state"] = overlay

        rows = self.window._japanese_mora_contributions(1)

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [row["role"] for row in rows],
            ["outgoing_vc", "incoming_cv"])

    def test_experimental_japanese_routing_controls_are_absent(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "aka", mode="kana")
        state = self.window.sentences[0]
        state["japanese_state"] = fg.je.new_edit_state(
            utterance, frontend_mode="kana")
        state["japanese_state"]["manual_candidate_overrides"] = {
            "0": "jc_manual"}
        panel = self.window.japanese_editor
        panel.set_state(state["japanese_state"])

        self.assertFalse(hasattr(panel, "dynamic_pitch"))
        self.assertFalse(hasattr(panel, "voice_color"))
        self.assertFalse(hasattr(panel, "question"))
        self.assertFalse(hasattr(panel, "boundary"))
        self.assertFalse(hasattr(panel, "inspect"))
        self.assertEqual(
            state["japanese_state"]["manual_candidate_overrides"],
            {"0": "jc_manual"})

    def test_mora_marker_double_click_places_nucleus_and_single_selects(self):
        from PyQt5 import QtTest

        utterance = fg.japanese_frontend.analyze_japanese(
            "\u304b\u306a", mode="kana")
        panel = self.window.japanese_editor
        overlay = fg.je.new_edit_state(utterance, frontend_mode="kana")
        self.window.sentences[0]["japanese_state"] = overlay
        panel.set_state(overlay)
        panel.grid.resize(500, 112)
        panel.grid.show()
        self.app.processEvents()
        events = []
        panel.grid.accentEdited.connect(
            lambda index, value: events.append((index, value)))
        rect = panel.grid._actual_cell_rect(1)

        QtTest.QTest.mouseClick(
            panel.grid, fg.Qt.LeftButton,
            pos=fg.QtCore.QPoint(int(rect.center().x()), 10))
        self.assertEqual(events, [])
        self.assertEqual(panel.grid.selected_mora, utterance.moras[1].index)
        QtTest.QTest.mouseDClick(
            panel.grid, fg.Qt.LeftButton,
            pos=fg.QtCore.QPoint(int(rect.center().x()), 10))
        self.assertEqual(events[-1][1], {
            "accent_state": "accented", "accent_nucleus": 1})
        self.app.processEvents()
        rect = panel.grid._actual_cell_rect(1)
        right_press = fg.QtGui.QMouseEvent(
            fg.QtCore.QEvent.MouseButtonPress,
            fg.QtCore.QPointF(rect.center().x(), 55),
            fg.Qt.RightButton, fg.Qt.RightButton, fg.Qt.NoModifier)
        panel.grid.mousePressEvent(right_press)
        self.assertEqual(events[-1][1]["accent_state"], "unaccented")

        overlay["accent_overrides"] = {
            "0": {"accent_state": "accented", "accent_nucleus": 0}}
        panel.set_state(overlay)
        events.clear()
        first = panel.grid._actual_cell_rect(0)
        second = panel.grid._actual_cell_rect(1)
        marker = fg.QtCore.QPointF(first.center().x(), first.top() + 7)
        target = fg.QtCore.QPointF(second.center().x(), second.top() + 7)
        panel.grid.mousePressEvent(fg.QtGui.QMouseEvent(
            fg.QtCore.QEvent.MouseButtonPress, marker,
            fg.Qt.LeftButton, fg.Qt.LeftButton, fg.Qt.NoModifier))
        panel.grid.mouseMoveEvent(fg.QtGui.QMouseEvent(
            fg.QtCore.QEvent.MouseMove, target,
            fg.Qt.NoButton, fg.Qt.LeftButton, fg.Qt.NoModifier))
        panel.grid.mouseReleaseEvent(fg.QtGui.QMouseEvent(
            fg.QtCore.QEvent.MouseButtonRelease, target,
            fg.Qt.LeftButton, fg.Qt.NoButton, fg.Qt.NoModifier))
        self.assertEqual(events[-1][1], {
            "accent_state": "accented", "accent_nucleus": 1})

    def test_phrase_pause_dialog_persists_and_marks_festival_text(self):
        state = self.window.sentences[0]
        state.update({
            "engine": "festival_wsl", "input_mode": "text",
            "rendered": True,
            "synthesis": fc.Synthesis(
                np.zeros(20, np.float32), 1000,
                [fc.Segment("pau", 0.0, .02)]),
        })
        with mock.patch.object(
                fg.PhrasePauseDialog, "exec_",
                return_value=fg.QtWidgets.QDialog.Accepted), \
                mock.patch.object(
                    fg.PhrasePauseDialog, "values",
                    return_value={
                        "minor": 150, "major": 350, "sentence": 650}):
            self.assertTrue(self.window.on_phrase_pauses())

        self.assertEqual(self.window.cfg["phrase_pauses_ms"], {
            "minor": 150, "major": 350, "sentence": 650})
        self.assertEqual(self.window._pending_action(state), "rerender")
        stored = json.loads(Path(fg.CONFIG_PATH).read_text(encoding="utf-8"))
        self.assertEqual(stored["phrase_pauses_ms"]["sentence"], 650)

    def test_ui_font_uses_points_and_has_japanese_fallback(self):
        font = fg.select_ui_font(10.0)

        self.assertAlmostEqual(font.pointSizeF(), 10.0)
        self.assertNotIn("font-size: 11px", fg.XP_QSS)
        self.assertIn("font-size: 9pt", fg.XP_QSS)
        supporting = []
        for family in fg.QtGui.QFontDatabase().families():
            candidate = fg.QtGui.QFont(family)
            candidate.setPointSizeF(10.0)
            if fg.font_has_japanese_glyphs(candidate):
                supporting.append(family)
        if supporting:
            self.assertTrue(fg.font_has_japanese_glyphs(font))
        else:
            self.assertTrue(font.family())

    def test_japanese_project_state_round_trips_and_old_rows_migrate(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "ながい？", mode="kana")
        state = self.window.sentences[0]
        state["text"] = "ながい？"
        state["japanese_state"] = fg.je.new_edit_state(
            utterance, frontend_mode="kana")
        state["japanese_state"]["mora_pitch_offsets_cents"] = {"1": 55}
        state["japanese_state"]["mora_voicing_overrides"] = {"0": 0.4}
        self.window.cfg["phrase_pauses_ms"] = {
            "minor": 135, "major": 315, "sentence": 525}

        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "Japanese Project"
            self.assertTrue(self.window.on_save_project(project))
            manifest = json.loads(
                (project / "project.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["sentences"][0]["japanese_state"]
                ["mora_pitch_offsets_cents"], {"1": 55})
            self.assertEqual(
                manifest["sentences"][0]["japanese_state"]
                ["mora_voicing_overrides"], {"0": 0.4})
            self.assertEqual(
                manifest["settings"]["phrase_pauses_ms"]["sentence"], 525)
            self.assertTrue(
                self.window.on_open_project(project / "project.json"))

        restored = self.window.sentences[0]["japanese_state"]
        self.assertEqual(restored["mora_pitch_offsets_cents"], {"1": 55})
        self.assertEqual(restored["mora_voicing_overrides"], {"0": 0.4})
        self.assertEqual(
            self.window.cfg["phrase_pauses_ms"]["major"], 315)
        migrated = self.window._state_from_project_row({
            "text": "かな",
            "japanese_state": {"mora_pitch_offsets": {0: 30}},
        })
        self.assertEqual(migrated["japanese_state"]
                         ["mora_pitch_offsets_cents"], {"0": 30})

    def test_asaxi_mora_state_round_trips_in_project(self):
        state = self.window.sentences[0]
        state["text"] = "shěso"
        state["lang_code"] = "asaxi"
        state["asaxi_state"] = fg.asaxi_editing.new_edit_state("shěso")
        state["asaxi_state"]["mora_tone_overrides"] = {"0": "L"}
        state["asaxi_state"]["mora_pitch_offsets_cents"] = {"0": 90}
        state["asaxi_state"]["mora_voicing_overrides"] = {"0": 0.3}

        with tempfile.TemporaryDirectory() as root:
            project = Path(root) / "Asaxi Project"
            self.assertTrue(self.window.on_save_project(project))
            manifest = json.loads(
                (project / "project.json").read_text(encoding="utf-8"))
            stored = manifest["sentences"][0]["asaxi_state"]
            self.assertEqual(
                stored["mora_tone_overrides"], {"0": "L"})
            self.assertEqual(
                stored["mora_pitch_offsets_cents"], {"0": 90})
            self.assertEqual(
                stored["mora_voicing_overrides"], {"0": 0.3})
            self.assertNotIn("mora_breathiness_overrides", stored)
            self.assertTrue(
                self.window.on_open_project(project / "project.json"))

        restored = self.window.sentences[0]["asaxi_state"]
        self.assertEqual(
            restored["mora_tone_overrides"], {"0": "L"})
        self.assertEqual(
            restored["mora_pitch_offsets_cents"], {"0": 90})
        self.assertEqual(
            restored["mora_voicing_overrides"], {"0": 0.3})

    def test_japanese_plan_uses_isolated_runtime_and_double_pauses(self):
        self.window.cfg["phrase_pauses_ms"] = dict(
            fg.fc.DEFAULT_PHRASE_PAUSES_MS)
        utterance = fg.japanese_frontend.analyze_japanese(
            "ねこ。いぬ？", mode="kana")
        state = {"japanese_state": fg.je.new_edit_state(
            utterance, frontend_mode="kana")}
        runtime = {
            "language": "ja",
            "voice_entry_point": "voice_test_ja",
            "candidate_units": {},
        }
        with mock.patch.object(
                self.window.fest, "japanese_runtime_metadata",
                return_value=runtime):
            plan = self.window._prepare_japanese_plan(
                state, utterance.source_text, "test", 1.0, 180.0,
                analyze=False)

        phones = plan.phones
        internal = next(index for index in range(2, len(phones) - 2)
                        if phones[index:index + 2] == ["pau", "pau"])
        self.assertGreater(internal, 1)
        self.assertEqual(state["japanese_state"]["last_plan"]["language"],
                         "ja")
        self.assertTrue(plan.f0_targets)
        fitted_gaps = [segment.duration for segment in plan.segments
                       if segment.pause_role == "phrase_gap"]
        fitted_guards = [segment.duration for segment in plan.segments
                         if segment.pause_role in {
                             "phrase_guard_out", "phrase_guard_in"}]
        self.assertEqual(len(fitted_gaps), 1)
        self.assertEqual(len(fitted_guards), 2)
        self.assertAlmostEqual(fitted_gaps[0], 0.72)
        self.assertAlmostEqual(sum(fitted_gaps + fitted_guards), 0.88)

        self.window.cfg["phrase_pauses_ms"] = {
            "minor": 135, "major": 315, "sentence": 525}
        with mock.patch.object(
                self.window.fest, "japanese_runtime_metadata",
                return_value=runtime):
            custom_plan = self.window._prepare_japanese_plan(
                state, utterance.source_text, "test", 1.0, 180.0,
                analyze=False)
        custom_gaps = [segment.duration for segment in custom_plan.segments
                       if segment.pause_role == "phrase_gap"]
        self.assertEqual(len(custom_gaps), 1)
        self.assertAlmostEqual(custom_gaps[0], 0.445)

    def test_japanese_question_rise_uses_general_intonation_blocks(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "これはテストですか？", mode="kana")
        plan = fg.je.create_edited_plan(
            utterance, fg.je.new_edit_state(
                utterance, frontend_mode="kana"),
            runtime_metadata={"language": "ja", "candidate_units": {}},
            base_pitch_hz=180.0,
        )
        rendered = mock.Mock()
        rendered.warning = None
        with mock.patch.object(
                self.window.fest, "synth_phones",
                return_value=rendered) as synth:
            self.window._render_japanese_plan(
                plan, "test", utterance.source_text,
                180.0, 18.0, False, {})

        kwargs = synth.call_args.kwargs
        self.assertEqual(kwargs["pitch_mode"], "intonation")
        self.assertEqual(kwargs["intonation_blocks"][-1]["kind"], "?")
        self.assertEqual(
            kwargs["ground_truth_targets"], list(plan.pitch_targets))
        self.assertGreater(
            kwargs["pitch_targets"][-1][1],
            kwargs["ground_truth_targets"][-1][1],
        )
        self.assertFalse(any(
            "interrogative" in target.kind for target in plan.f0_targets
        ))

    def test_japanese_zero_fall_statement_uses_generated_f0_only(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "ねこです。", mode="kana")
        plan = fg.je.create_edited_plan(
            utterance, fg.je.new_edit_state(
                utterance, frontend_mode="kana"),
            runtime_metadata={"language": "ja", "candidate_units": {}},
            base_pitch_hz=213.0,
        )
        rendered = mock.Mock()
        rendered.warning = None
        with mock.patch.object(
                self.window.fest, "synth_phones",
                return_value=rendered) as synth:
            self.window._render_japanese_plan(
                plan, "test", utterance.source_text,
                213.0, 0.0, False, {})

        kwargs = synth.call_args.kwargs
        self.assertEqual(kwargs["pitch_mode"], "")
        self.assertIsNone(kwargs["intonation_blocks"])
        self.assertEqual(kwargs["pitch_targets"], list(plan.pitch_targets))
        self.assertEqual(
            kwargs["ground_truth_targets"], list(plan.pitch_targets))

    def test_japanese_render_forwards_mora_voicing_overrides(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "\u304d\u304f", mode="kana")
        plan = fg.je.create_edited_plan(
            utterance, fg.je.new_edit_state(utterance),
            runtime_metadata={"language": "ja", "candidate_units": {}},
        )
        segments = fc.segments_from_durations(plan.segment_durations)
        rendered = fc.Synthesis(
            np.zeros(max(1, round(segments[-1].end * 16000)), np.float32),
            16000, segments, lang="ja")
        overrides = {0: 0.35, 1: 0.55}
        with mock.patch.object(
                self.window.fest, "synth_phones",
                return_value=rendered), mock.patch.object(
                    fg.japanese_devoicing, "apply_vowel_realizations",
                    return_value=rendered) as apply:
            self.window._render_japanese_plan(
                plan, "test", utterance.source_text,
                180.0, 18.0, False, {},
                mora_voicing_overrides=overrides)

            apply.assert_not_called()
            self.window._apply_shared_voicing_stage(rendered)

        self.assertEqual(
            apply.call_args.kwargs["mora_voicing_overrides"], overrides)
        self.assertEqual(rendered.japanese_prosody["duration_model"],
                         plan.duration_model)
        self.assertEqual(rendered.japanese_prosody["duration_model_id"],
                         plan.duration_model_id)
        self.assertEqual(rendered.japanese_prosody["pitch_model_id"],
                         plan.pitch_model_id)
        self.assertFalse(rendered.japanese_prosody[
            "cumulative_register_drift_enabled"])
        self.assertEqual(rendered.japanese_prosody[
            "phrase_position_model"], "mean_centered_shape")
        self.assertAlmostEqual(
            rendered.japanese_prosody["total_duration_seconds"],
            sum(duration for _phone, duration in plan.segment_durations),
            places=5,
        )
        self.assertGreater(
            rendered.japanese_prosody["f0_span_semitones"], 0.0)

    def test_asaxi_render_forwards_mora_phonation_overrides(self):
        segments = [
            fc.Segment("sh", 0.0, 0.05),
            fc.Segment("er", 0.05, 0.13),
            fc.Segment("s", 0.13, 0.18),
        ]
        metadata = {
            "moras": [{
                "mora_index": 0,
                "phrase_index": 0,
                "word": "shěso",
                "text": "shě",
                "phones": ["sh", "er"],
                "segment_indices": [0, 1],
            }]
        }
        rendered = fc.Synthesis(
            np.zeros(round(0.18 * 16000), np.float32),
            16000, segments, lang="asaxi")
        voicing = {"0": 0.25}
        continuous = [(0.0, 1.0), (0.18, 0.8)]
        self.window._set_language_render_features(
            rendered,
            asaxi_metadata=metadata,
            voicing_override=continuous,
            asaxi_voicing_overrides=voicing,
        )
        with mock.patch.object(
                fg.asaxi_phonation, "apply_phonation",
                return_value=rendered) as apply:
            self.window._apply_shared_voicing_stage(rendered)

        kwargs = apply.call_args.kwargs
        self.assertEqual(kwargs["voicing_overrides"], voicing)
        self.assertNotIn("breathiness_overrides", kwargs)
        self.assertEqual(
            kwargs["continuous_voicing_override"], continuous)
        self.assertEqual(apply.call_args.args[1], metadata)

    def test_generated_wsl_languages_share_post_render_level_policy(self):
        metadata = {
            "kind": "festival_unisyn_runtime_index",
            "voice_manifest_schema_version": 1,
            "source_bundle_id": "srb_fixture",
            "configuration_id": "vcfg_fixture",
            "builder_version": "unified-festival-builder-v1",
        }
        backend = mock.Mock()
        backend.voice_metadata.return_value = metadata
        segment = [fc.Segment("a", 0.0, 0.2)]
        english = fc.Synthesis(
            np.ones(200, np.float32) * 0.072, 1000, segment,
            lang="en", voicebank="fixture")
        japanese = fc.Synthesis(
            np.ones(200, np.float32) * 0.032, 1000, segment,
            lang="ja", voicebank="fixture")

        with mock.patch.object(self.window, "_ab", return_value=backend):
            self.window._apply_voice_output_calibration(english, "fixture")
            self.window._apply_voice_output_calibration(japanese, "fixture")

        english_rms, _ = fc.active_speech_rms(
            english.samples, english.sr, english.segments)
        japanese_rms, _ = fc.active_speech_rms(
            japanese.samples, japanese.sr, japanese.segments)
        self.assertAlmostEqual(english_rms, 0.1, places=5)
        self.assertAlmostEqual(japanese_rms, 0.1, places=5)
        self.assertGreater(japanese.automatic_gain_db,
                           english.automatic_gain_db)
        self.assertEqual(
            japanese.output_calibration["policy_source"],
            "legacy_generated_voice_default")

    def test_all_languages_use_same_final_acoustic_stage_order(self):
        orders = {}
        for language in ("en", "asaxi", "ja"):
            calls = []
            syn = fc.Synthesis(
                np.ones(200, np.float32) * 0.05, 1000,
                [fc.Segment("a", 0.0, 0.2)], lang=language,
                voicebank="fixture")
            with mock.patch.object(
                    self.window, "_apply_shared_voicing_stage",
                    side_effect=lambda value: (
                        calls.append("voicing") or value)), \
                    mock.patch.object(
                        self.window, "_apply_vocal_tract_transform",
                        side_effect=lambda value, *_args, **_kwargs: (
                            calls.append("vocal_tract") or value)), \
                    mock.patch.object(
                        self.window, "_apply_voice_output_calibration",
                        side_effect=lambda value, *_args, **_kwargs: (
                            calls.append("output_calibration") or value)):
                self.window._apply_output_faults(syn, faults={})
            orders[language] = calls

        expected = ["voicing", "vocal_tract", "output_calibration"]
        self.assertEqual(orders, {
            "en": expected,
            "asaxi": expected,
            "ja": expected,
        })

    def test_unknown_festival_voice_does_not_receive_generated_policy(self):
        backend = mock.Mock()
        backend.voice_metadata.return_value = {}
        syn = fc.Synthesis(
            np.ones(200, np.float32) * 0.02, 1000,
            [fc.Segment("a", 0.0, 0.2)], voicebank="external")

        with mock.patch.object(self.window, "_ab", return_value=backend):
            self.window._apply_voice_output_calibration(syn, "external")

        self.assertEqual(syn.output_calibration, {})
        self.assertEqual(syn.automatic_gain_db, 0.0)

    def test_japanese_rerender_preserves_editor_durations_exactly(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "kore wa tesuto desu.", mode="kana")
        plan = fg.je.create_edited_plan(
            utterance, fg.je.new_edit_state(utterance),
            runtime_metadata={"language": "ja", "candidate_units": {}},
        )
        edited_entries = list(plan.segment_durations)
        edited_entries[2] = (
            edited_entries[2][0], edited_entries[2][1] * 1.7)
        edited_entries[3] = (
            edited_entries[3][0], edited_entries[3][1] * 0.65)
        segments = fc.segments_from_durations(edited_entries)
        current = fc.Synthesis(
            np.zeros(max(1, int(segments[-1].end * 1000)), np.float32),
            1000, segments, text=utterance.source_text, lang="ja",
            voicebank="test", targets=list(plan.pitch_targets),
            generated_targets=list(plan.pitch_targets))
        self.window._show_synthesis(current)
        expected_entries = self.window._rerender_seg_durs()
        self.window.sentences[0]["japanese_state"] = \
            fg.je.new_edit_state(utterance)
        self.window._clear_state_pending(self.window.sentences[0])
        captured = {}

        class FakeBackend:
            def synth_phones(_self, phones, voicebank, speed=1.0, **kwargs):
                captured.update(kwargs)
                rendered = fc.segments_from_durations(kwargs["seg_durs"])
                return fc.Synthesis(
                    np.zeros(max(1, int(rendered[-1].end * 1000)),
                             np.float32),
                    1000, rendered, text=kwargs.get("text", ""), lang="ja",
                    voicebank=voicebank, targets=list(
                        kwargs.get("ground_truth_targets") or []),
                    generated_targets=list(
                        kwargs.get("ground_truth_targets") or []))

        self.window._need_backend = lambda: True
        self.window._ab = lambda: FakeBackend()
        self.window._current_voicebank = lambda: "test"
        self.window.input_mode.setCurrentIndex(
            self.window.input_mode.findData("text"))
        with mock.patch.object(
                self.window, "_engine", return_value="festival_wsl"), \
                mock.patch.object(
                    self.window, "_current_lang_code", return_value="ja"), \
                mock.patch.object(
                    self.window, "_prepare_japanese_plan",
                    return_value=plan), \
                mock.patch.object(
                    self.window, "_refresh_voice_metadata"), \
                mock.patch.object(
                    fg.japanese_devoicing, "apply_vowel_realizations"):
            self.window.on_rerender()

        self.assertEqual(
            [phone for phone, _duration in captured["seg_durs"]],
            [phone for phone, _duration in expected_entries])
        np.testing.assert_allclose(
            [duration for _phone, duration in captured["seg_durs"]],
            [duration for _phone, duration in expected_entries],
            rtol=0, atol=1e-12)
        self.assertNotAlmostEqual(
            captured["seg_durs"][2][1], plan.segment_durations[2][1])

    def test_render_details_exposes_japanese_mora_timing_safety(self):
        utterance = fg.japanese_frontend.analyze_japanese(
            "きゃく。", mode="kana")
        plan = fg.je.create_edited_plan(
            utterance, fg.je.new_edit_state(utterance),
            runtime_metadata={"language": "ja", "candidate_units": {}},
        )
        self.window.sentences[0]["japanese_state"] = {
            "utterance": utterance.to_dict(),
            "last_plan": plan.to_dict(),
        }
        self.window.current = fc.Synthesis(
            np.zeros(100, np.float32), 16000,
            text=utterance.source_text, lang="ja", voicebank="test",
            phones=plan.phones,
            vowel_realizations=[{
                "segment_index": 2, "mora_index": 0, "phone": "u",
                "strategy": "source_filter_residual_devoiced",
                "reason": "periodicity fell",
                "periodicity_before": .82, "periodicity_after": .25,
            }],
        )
        with mock.patch.object(
                self.window, "_current_lang_code", return_value="ja"), \
                mock.patch.object(
                    fg.QtWidgets.QMessageBox, "information") as information:
            self.window.on_render_details()

        message = information.call_args.args[2]
        self.assertIn("Japanese active prosody models", message)
        self.assertIn(plan.duration_model_id, message)
        self.assertIn(plan.pitch_model_id, message)
        self.assertIn("timeline:", message)
        self.assertIn("cumulative register drift: disabled", message)
        self.assertIn("Japanese mora timing / source safety", message)
        self.assertIn("predicted", message)
        self.assertIn("safe", message)
        self.assertIn("Japanese vowel realization", message)
        self.assertIn("source_filter_residual_devoiced", message)

    def test_japanese_plan_does_not_apply_old_occurrence_edits_to_new_text(self):
        old = fg.japanese_frontend.analyze_japanese("neko", mode="kana")
        new = fg.japanese_frontend.analyze_japanese("inu", mode="kana")
        overlay = fg.je.new_edit_state(old, frontend_mode="kana")
        overlay["mora_pitch_offsets_cents"] = {"0": 200}
        overlay["manual_candidate_overrides"] = {"0": "jc_stale"}
        overlay["profile_path"] = "profile.json"
        state = {"japanese_state": overlay}
        runtime = {
            "language": "ja",
            "voice_entry_point": "voice_test_ja",
            "candidate_units": {},
        }

        with mock.patch.object(
                fg.japanese_frontend, "analyze_japanese",
                return_value=new), mock.patch.object(
                self.window.fest, "japanese_runtime_metadata",
                return_value=runtime):
            self.window._prepare_japanese_plan(
                state, new.source_text, "test", 1.0, 180.0,
                analyze=True)

        result = state["japanese_state"]
        self.assertEqual(result["mora_pitch_offsets_cents"], {})
        self.assertEqual(result["manual_candidate_overrides"], {})
        self.assertEqual(result["profile_path"], "profile.json")
        self.assertEqual(result["utterance"]["source_text"], new.source_text)

    def test_japanese_bank_dialog_displays_coverage_and_unresolved_rows(self):
        with tempfile.TemporaryDirectory() as root:
            bank = Path(root) / "bank"
            bank.mkdir()
            fc.write_wav(bank / "a.wav", np.zeros(4410, np.float32), 44100)
            fc.write_wav(
                bank / "mystery.wav", np.zeros(4410, np.float32), 44100)
            (bank / "oto.ini").write_text(
                "a.wav=a,0,80,-100,40,20\n"
                "mystery.wav=??? token,0,80,-100,40,20\n",
                encoding="utf-8")
            analysis = fg.je.analyze_bank(bank)
            dialog = fg.JapaneseBankAnalysisDialog(
                analysis, parent=self.window)

            self.assertIn("source entries", dialog.coverage.text())
            self.assertGreaterEqual(dialog.table.rowCount(), 1)
            dialog.configuration.setCurrentIndex(
                dialog.configuration.findData("mixed"))
            self.assertTrue(dialog.profile_changed)
            self.assertEqual(dialog.profile.bank_configuration, "mixed")
            dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()
