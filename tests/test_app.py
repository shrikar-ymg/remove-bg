"""Focused route tests that do not download or run the segmentation models."""

import io
import os
import unittest

# Set before the app loads .env (python-dotenv never overrides an existing
# variable), so route tests cannot write to a real MongoDB cluster.
os.environ["MONGODB_URI"] = ""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app import (
    background_profile,
    fill_graphic_interiors,
    APP_VERSION,
    AUTO_PHOTO_CANDIDATES,
    AUTO_PREFERRED_QUALITY,
    AUTO_QUALITY,
    AUTO_TIE_MARGIN,
    CF_SOLVE_MAX_PIXELS,
    GRABCUT_MAX_PIXELS,
    app,
    available_provider_priority,
    combine_masks,
    detect_natural_shadow,
    directml_is_suitable,
    finish_cutout,
    provider_chain,
    choose_automatic_mask,
    mask_edge_agreement,
    recover_all_elements,
    smart_refine,
    solve_alpha_bounded,
    preserve_natural_shadow,
    runtime_backend_info,
    session_options,
)


def cutout_bytes() -> bytes:
    image = Image.new("RGBA", (2, 2), (220, 40, 30, 0))
    image.putpixel((1, 0), (220, 40, 30, 128))
    image.putpixel((0, 1), (20, 180, 90, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def synthetic_product(with_shadow=True):
    size = (320, 240)
    background = Image.new("RGBA", size, (244, 244, 240, 255))
    if with_shadow:
        shadow = Image.new("RGBA", size, (0, 0, 0, 0))
        shadow_mask = Image.new("L", size, 0)
        ImageDraw.Draw(shadow_mask).ellipse((68, 148, 256, 211), fill=72)
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(9))
        shadow.putalpha(shadow_mask)
        background = Image.alpha_composite(background, shadow)

    ImageDraw.Draw(background).rounded_rectangle(
        (112, 48, 208, 174),
        radius=10,
        fill=(186, 53, 38, 255),
    )
    subject_mask = Image.new("L", size, 0)
    ImageDraw.Draw(subject_mask).rounded_rectangle(
        (112, 48, 208, 174),
        radius=10,
        fill=255,
    )
    cutout = Image.new("RGBA", size, (186, 53, 38, 0))
    cutout.putalpha(subject_mask)
    return background, cutout, subject_mask


def flat_graphic(badge_fill=(20, 90, 230), gap_shows_page=True):
    """Build an illustration: a flat page, a ringed badge, and a see-through gap.

    This is the shape of the case a saliency model gets wrong - the badge's
    coloured disc shares the page's hue, so the model keeps the white ring and
    drops the fill inside it.
    """
    page = (31, 114, 253)
    image = Image.new("RGB", (240, 240), page)
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 140, 140), fill=badge_fill)          # the disc
    draw.ellipse((40, 40, 140, 140), outline=(255, 255, 255), width=7)
    draw.ellipse((70, 70, 110, 110), fill=(255, 255, 255))     # the glyph
    # A handle-like shape whose opening genuinely shows the page through.
    draw.ellipse((150, 150, 220, 220), fill=(240, 240, 245))
    if gap_shows_page:
        draw.ellipse((170, 170, 200, 200), fill=page)

    # The mask a saliency model produces: ring and glyph kept, disc dropped.
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((40, 40, 140, 140), outline=255, width=7)
    mask_draw.ellipse((70, 70, 110, 110), fill=255)
    mask_draw.ellipse((150, 150, 220, 220), fill=255)
    if gap_shows_page:
        mask_draw.ellipse((170, 170, 200, 200), fill=0)
    return image.convert("RGBA"), mask


class AppRouteTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def export(self, output_format="png", background="transparent", color="#ffffff"):
        return self.client.post(
            "/export",
            data={
                "current": (io.BytesIO(cutout_bytes()), "current.png"),
                "format": output_format,
                "filename": "sample.photo.png",
                "background": background,
                "background_color": color,
            },
            content_type="multipart/form-data",
        )

    def test_home_and_product_info(self):
        home = self.client.get("/")
        info = self.client.get("/api/info")
        health = self.client.get("/health")

        self.assertEqual(home.status_code, 200)
        self.assertIn(b"Cutout Studio 2.2", home.data)
        self.assertEqual(info.json["version"], APP_VERSION)
        self.assertIn("best", info.json["quality_profiles"])
        self.assertFalse(info.json["quality_profiles"]["best"]["preserve_shadows"])
        self.assertTrue(info.json["runtime"]["automatic"])
        self.assertEqual(health.json["status"], "ok")
        self.assertEqual(home.headers["X-Content-Type-Options"], "nosniff")

    def test_runtime_configuration_has_cpu_fallback(self):
        priority = available_provider_priority()
        self.assertIn("CPUExecutionProvider", priority)
        self.assertEqual(provider_chain("CPUExecutionProvider"), ["CPUExecutionProvider"])
        self.assertEqual(provider_chain("DmlExecutionProvider")[-1], "CPUExecutionProvider")
        self.assertFalse(session_options("DmlExecutionProvider").enable_mem_pattern)
        self.assertTrue(runtime_backend_info()["automatic"])
        self.assertFalse(
            directml_is_suitable(
                {
                    "name": "NVIDIA GeForce MX230",
                    "memory_mb": 1968,
                    "discrete": True,
                }
            )
        )
        self.assertTrue(
            directml_is_suitable(
                {
                    "name": "AMD Radeon RX 7600",
                    "memory_mb": 8192,
                    "discrete": True,
                }
            )
        )

    def test_transparent_png_keeps_alpha(self):
        response = self.export()
        self.assertEqual(response.status_code, 200)
        with Image.open(io.BytesIO(response.data)) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getpixel((0, 0))[3], 0)

    def test_custom_background_is_composited(self):
        response = self.export(background="custom", color="#123456")
        self.assertEqual(response.status_code, 200)
        with Image.open(io.BytesIO(response.data)) as image:
            self.assertEqual(image.convert("RGB").getpixel((0, 0)), (18, 52, 86))

    def test_jpeg_uses_white_when_preview_is_transparent(self):
        response = self.export(output_format="jpg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        with Image.open(io.BytesIO(response.data)) as image:
            self.assertEqual(image.mode, "RGB")
            red, green, blue = image.getpixel((0, 0))
            self.assertGreater(red, 235)
            self.assertGreater(green, 235)
            self.assertGreater(blue, 235)

    def test_svg_embeds_the_current_image(self):
        response = self.export(output_format="svg", background="black")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<svg xmlns="http://www.w3.org/2000/svg"', response.data)
        self.assertIn(b"data:image/png;base64,", response.data)

    def test_invalid_export_options_are_rejected(self):
        response = self.export(background="custom", color="not-a-color")
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid custom background", response.json["error"])

    def test_invalid_shadow_option_is_rejected_before_inference(self):
        response = self.client.post(
            "/remove",
            data={
                "image": (io.BytesIO(cutout_bytes()), "sample.png"),
                "quality": "best",
                "preserve_shadows": "sometimes",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("shadow preservation", response.json["error"])

    def test_shadow_detection_keeps_contact_shadow_but_not_flat_background(self):
        original, _cutout, subject_mask = synthetic_product(with_shadow=True)
        detected = np.asarray(detect_natural_shadow(original, subject_mask))
        flat, _cutout, flat_mask = synthetic_product(with_shadow=False)
        flat_detected = np.asarray(detect_natural_shadow(flat, flat_mask))

        self.assertGreater(float(np.mean(detected[178:208, 85:240])), 4.0)
        self.assertLess(float(np.mean(detected[:35])), 0.2)
        self.assertLess(float(np.mean(flat_detected)), 0.35)

    def test_shadow_compositing_does_not_change_subject_pixels(self):
        original, cutout, _subject_mask = synthetic_product(with_shadow=True)
        preserved = preserve_natural_shadow(original, cutout)

        self.assertEqual(preserved.getpixel((160, 100)), cutout.getpixel((160, 100)))
        shadow_pixel = preserved.getpixel((160, 193))
        self.assertGreater(shadow_pixel[3], 6)
        self.assertEqual(shadow_pixel[:3], (0, 0, 0))

    def test_precision_finishing_keeps_a_compact_edge_transition(self):
        size = (128, 96)
        original = Image.new("RGBA", size, (245, 245, 242, 255))
        ImageDraw.Draw(original).rectangle((36, 20, 92, 78), fill=(30, 105, 205, 255))
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle((36, 20, 92, 78), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(1.4))

        result = finish_cutout(original, mask, max_side=128, precision=True)
        alpha = np.asarray(result.getchannel("A"))
        transition = np.count_nonzero((alpha[48] > 12) & (alpha[48] < 243))

        self.assertLessEqual(transition, 10)
        self.assertEqual(int(alpha[48, 64]), 255)
        self.assertEqual(int(alpha[48, 5]), 0)

    def test_precision_finishing_preserves_low_contrast_transparency(self):
        width, height = 160, 80
        original = Image.new("RGBA", (width, height), (132, 135, 138, 255))
        gradient = np.tile(
            np.linspace(0, 255, width, dtype=np.uint8),
            (height, 1),
        )
        mask = Image.fromarray(gradient, mode="L")

        result = finish_cutout(
            original,
            mask,
            max_side=160,
            precision=True,
        )
        finished = np.asarray(result.getchannel("A"), dtype=np.int16)

        self.assertLess(float(np.mean(np.abs(finished - gradient))), 3.0)

    def test_graphic_recovery_never_erases_the_ai_mask(self):
        size = (240, 180)
        original = Image.new("RGB", size, (28, 92, 190))
        draw = ImageDraw.Draw(original)
        draw.rectangle((20, 35, 90, 145), fill=(235, 240, 248))
        draw.ellipse((155, 45, 215, 105), fill=(245, 245, 248))

        ai = Image.new("L", size, 0)
        ai_draw = ImageDraw.Draw(ai)
        ai_draw.rectangle((20, 35, 90, 145), fill=255)
        # A BEN2-only detail deliberately omitted by color recovery.
        ai_draw.rectangle((110, 130, 145, 160), fill=190)

        recovered = Image.new("L", size, 0)
        recovered_draw = ImageDraw.Draw(recovered)
        recovered_draw.ellipse((155, 45, 215, 105), fill=255)
        combined = np.asarray(
            combine_masks(original, ai, recovered),
            dtype=np.uint8,
        )
        ai_array = np.asarray(ai, dtype=np.uint8)

        self.assertTrue(np.all(combined >= ai_array))
        self.assertGreater(int(combined[75, 185]), 200)

    def test_smooth_graphic_recovery_keeps_inner_icons_and_dashes(self):
        width, height = 320, 240
        x_gradient = np.linspace(0, 22, width, dtype=np.uint8)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = 28 + x_gradient
        rgb[:, :, 1] = 104 + x_gradient
        rgb[:, :, 2] = 220 + np.minimum(x_gradient, 30)
        original = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(original)
        draw.rounded_rectangle((32, 45, 145, 205), radius=18, fill=(235, 240, 248))
        draw.ellipse((220, 42, 292, 114), fill=(238, 242, 250))
        draw.rectangle((242, 63, 270, 94), fill=(20, 54, 116))
        for x in range(153, 214, 14):
            draw.line((x, 78, min(x + 7, 214), 78), fill=(242, 245, 250), width=3)

        semantic = Image.new("L", (width, height), 0)
        semantic_draw = ImageDraw.Draw(semantic)
        semantic_draw.rounded_rectangle((32, 45, 145, 205), radius=18, fill=255)
        semantic_draw.ellipse((220, 42, 292, 114), fill=255)
        semantic_draw.rectangle((240, 61, 272, 96), fill=0)

        recovered = recover_all_elements(original)
        combined = np.maximum(
            np.asarray(semantic, dtype=np.uint8),
            np.asarray(recovered, dtype=np.uint8),
        )

        self.assertGreater(int(combined[78, 159]), 96)
        self.assertGreater(int(combined[78, 256]), 192)
        self.assertLess(int(combined[20, 160]), 12)

    def test_shadow_composites_behind_a_weak_model_response(self):
        original, cutout, _subject_mask = synthetic_product(with_shadow=True)
        weak = np.asarray(cutout).copy()
        weak[193, 160] = (225, 225, 220, 18)
        preserved = preserve_natural_shadow(original, Image.fromarray(weak))

        red, green, blue, alpha = preserved.getpixel((160, 193))
        self.assertGreater(alpha, 18)
        self.assertLess(red, 225)
        self.assertLess(green, 225)
        self.assertLess(blue, 220)


class GraphicInteriorTests(unittest.TestCase):
    """A flat illustration is a colour problem, not a saliency problem."""

    def test_flat_illustration_is_recognized(self):
        image, _mask = flat_graphic()
        profile = background_profile(image)
        self.assertIsNotNone(profile)
        self.assertGreater(profile["flatness"], 0.65)
        np.testing.assert_allclose(profile["color"], (31, 114, 253), atol=6)

    def test_photograph_is_not_treated_as_a_flat_illustration(self):
        # Photographic noise and a busy border are what disqualify an image.
        rng = np.random.default_rng(4)
        noisy = rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)
        self.assertIsNone(background_profile(Image.fromarray(noisy, "RGB")))

        gradient = np.linspace(0, 255, 240, dtype=np.uint8)
        busy = np.dstack([
            np.tile(gradient, (240, 1)),
            np.tile(gradient[::-1], (240, 1)),
            np.tile(gradient, (240, 1)).T,
        ])
        self.assertIsNone(background_profile(Image.fromarray(busy, "RGB")))

    def test_badge_fill_is_restored_but_see_through_gap_is_not(self):
        image, mask = flat_graphic()
        profile = background_profile(image)
        filled = np.asarray(fill_graphic_interiors(image, mask, profile))
        before = np.asarray(mask)

        # The disc between ring and glyph is enclosed and unlike the page, so it
        # comes back.
        self.assertLess(int(before[95, 55]), 128)
        self.assertGreaterEqual(int(filled[95, 55]), 128)
        # The handle opening shows the page through and must stay open.
        self.assertLess(int(filled[185, 185]), 128)

    def test_filling_never_reaches_outside_the_subject(self):
        image, mask = flat_graphic()
        profile = background_profile(image)
        filled = np.asarray(fill_graphic_interiors(image, mask, profile))
        # Open page far from any shape is not enclosed, so it cannot be filled.
        for point in ((10, 10), (230, 10), (10, 230), (120, 200)):
            self.assertLess(int(filled[point]), 128, f"page filled at {point}")

    def test_a_hole_matching_the_page_colour_stays_open(self):
        image, mask = flat_graphic(badge_fill=(31, 114, 253))
        profile = background_profile(image)
        filled = np.asarray(fill_graphic_interiors(image, mask, profile))
        # The disc is now exactly the page colour, so it reads as see-through.
        self.assertLess(int(filled[95, 55]), 128)


class BoundedSolveTests(unittest.TestCase):
    """A large image must not be able to ask for an unbounded allocation.

    Closed-form matting holds 25 doubles per pixel in each of two arrays, so an
    unbounded solve on a 10-megapixel photograph asks for about 4 GB and fails
    outright. Both callers - background removal and the correction brush, whose
    region is the stroke's bounding box - are capped instead.
    """

    def trimap_scene(self, width, height):
        rgb = np.zeros((height, width, 3), np.float32)
        rgb[:, : width // 2] = (0.85, 0.85, 0.82)
        rgb[:, width // 2 :] = (0.15, 0.22, 0.40)
        trimap = np.full((height, width), 0.5, np.float32)
        trimap[:, : width // 4] = 0.0
        trimap[:, -width // 4 :] = 1.0
        return rgb, trimap

    def test_oversized_solve_is_reduced_not_refused(self):
        # Comfortably past the cap, so the reduced path is what runs.
        side = int((CF_SOLVE_MAX_PIXELS * 2) ** 0.5)
        rgb, trimap = self.trimap_scene(side, side)
        solved = solve_alpha_bounded(rgb, trimap, maxiter=20)

        self.assertEqual(solved.shape, (side, side))
        self.assertGreaterEqual(float(solved.min()), 0.0)
        self.assertLessEqual(float(solved.max()), 1.0)
        # The committed ends of the trimap must survive the round trip.
        self.assertLess(float(solved[:, : side // 8].mean()), 0.25)
        self.assertGreater(float(solved[:, -side // 8 :].mean()), 0.75)

    def test_small_solve_keeps_full_resolution(self):
        rgb, trimap = self.trimap_scene(96, 72)
        solved = solve_alpha_bounded(rgb, trimap, maxiter=20)
        self.assertEqual(solved.shape, (72, 96))

    def test_degenerate_input_returns_a_usable_matte(self):
        rgb = np.zeros((64, 64, 3), np.float32)
        trimap = np.full((64, 64), 0.5, np.float32)
        solved = solve_alpha_bounded(rgb, trimap, maxiter=10)
        self.assertEqual(solved.shape, (64, 64))
        self.assertTrue(np.all(np.isfinite(solved)))

    def test_long_stroke_over_a_large_image_completes(self):
        # A stroke dragged corner to corner is what made the region as large as
        # the picture itself.
        side = int((GRABCUT_MAX_PIXELS * 1.8) ** 0.5)
        source = Image.new("RGBA", (side, side), (210, 205, 195, 255))
        ImageDraw.Draw(source).ellipse(
            (side // 5, side // 5, side * 4 // 5, side * 4 // 5),
            fill=(40, 70, 130, 255),
        )
        points = [
            [round(side * 0.05 + side * 0.9 * step / 20),
             round(side * 0.05 + side * 0.9 * step / 20)]
            for step in range(21)
        ]
        result = smart_refine(source, source.copy(), points, 48, "erase")
        self.assertEqual(result.size, (side, side))
        self.assertEqual(result.mode, "RGBA")

    def test_out_of_memory_is_reported_as_a_capacity_problem(self):
        import app as app_module

        original = app_module.create_cutout

        def raise_memory_error(*args, **kwargs):
            raise MemoryError("Unable to allocate 1.86 GiB")

        app_module.create_cutout = raise_memory_error
        try:
            response = app.test_client().post(
                "/remove",
                data={
                    "image": (io.BytesIO(cutout_bytes()), "sample.png"),
                    "quality": "balanced",
                },
                content_type="multipart/form-data",
            )
        finally:
            app_module.create_cutout = original

        self.assertEqual(response.status_code, 507)
        self.assertIn("memory", response.json["error"].lower())
        self.assertNotIn("Traceback", response.json["error"])


class AutomaticSelectionTests(unittest.TestCase):
    """Automatic mode has to pick a model without a person judging the result.

    The measurement it uses is how much of a matte's own boundary sits on a real
    edge in the picture. On a traffic photograph where one model kept a second,
    out-of-focus vehicle this ranked the correct model first by a 34% margin;
    on a wispy-hair scene the two leaders were within 0.02%, which is why a
    near-tie defers to the general-purpose model rather than to noise.
    """

    def edge_scene(self):
        """A photograph-like scene, so routing sends it to the comparison.

        Grain is what separates a photograph from a flat illustration here; a
        noiseless version of this scene is routed straight to one model instead.
        """
        image = Image.new("RGB", (240, 180), (232, 230, 224))
        ImageDraw.Draw(image).rectangle((60, 40, 180, 140), fill=(40, 60, 120))
        grain = np.random.default_rng(4).normal(0, 9, (180, 240, 3))
        noisy = np.clip(np.asarray(image, np.float32) + grain, 0, 255)
        return Image.fromarray(noisy.astype(np.uint8)).convert("RGBA")

    def test_the_photograph_scene_is_not_routed_as_a_graphic(self):
        from app import background_profile as profile_of

        self.assertIsNone(profile_of(self.edge_scene()))

    def mask_from_box(self, box, size=(240, 180)):
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle(box, fill=255)
        return mask

    def test_boundary_on_a_real_edge_scores_above_one_that_is_not(self):
        image = self.edge_scene()
        aligned = self.mask_from_box((60, 40, 180, 140))
        # Same area, but its boundary runs through flat background instead.
        adrift = self.mask_from_box((20, 15, 140, 115))

        self.assertGreater(
            mask_edge_agreement(image, aligned),
            mask_edge_agreement(image, adrift),
        )

    def test_an_empty_or_full_matte_scores_zero(self):
        image = self.edge_scene()
        self.assertEqual(mask_edge_agreement(image, Image.new("L", (240, 180), 0)), 0.0)
        self.assertEqual(mask_edge_agreement(image, Image.new("L", (240, 180), 255)), 0.0)

    def test_score_is_comparable_between_high_and_low_contrast(self):
        # Normalising by a high percentile keeps a soft picture from scoring low
        # merely because every edge in it is gentle.
        strong = self.edge_scene()
        faint = Image.new("RGB", (240, 180), (150, 150, 150))
        ImageDraw.Draw(faint).rectangle((60, 40, 180, 140), fill=(140, 142, 148))
        aligned = self.mask_from_box((60, 40, 180, 140))
        self.assertAlmostEqual(
            mask_edge_agreement(strong, aligned),
            mask_edge_agreement(faint.convert("RGBA"), aligned),
            delta=0.25,
        )

    def test_illustrations_route_without_running_a_comparison(self):
        # A flat page goes straight to the precision model: it measured best or
        # joint-best on every illustration tested, so a second inference would
        # only cost time.
        page, _mask = flat_graphic()
        calls = []
        import app as app_module

        original = app_module.predict_model_mask

        def counting_predict(image, quality):
            calls.append(quality)
            return Image.new("L", image.size, 255)

        app_module.predict_model_mask = counting_predict
        try:
            quality, mask, graphic = choose_automatic_mask(page)
        finally:
            app_module.predict_model_mask = original

        self.assertEqual(quality, AUTO_PREFERRED_QUALITY)
        self.assertEqual(calls, [AUTO_PREFERRED_QUALITY])
        self.assertIsNotNone(graphic)

    def test_a_clear_winner_is_taken_over_the_preferred_model(self):
        image = self.edge_scene()
        import app as app_module

        original = app_module.predict_model_mask
        # The non-preferred candidate is given the boundary that follows the
        # picture, so the comparison has to override the default preference.
        masks = {
            "best": self.mask_from_box((20, 15, 140, 115)),
            "balanced": self.mask_from_box((60, 40, 180, 140)),
        }
        app_module.predict_model_mask = lambda img, quality: masks[quality]
        try:
            quality, mask, graphic = choose_automatic_mask(image)
        finally:
            app_module.predict_model_mask = original

        self.assertEqual(quality, "balanced")
        self.assertIsNone(graphic)

    def test_a_near_tie_defers_to_the_preferred_model(self):
        image = self.edge_scene()
        import app as app_module

        original = app_module.predict_model_mask
        shared = self.mask_from_box((60, 40, 180, 140))
        # Identical mattes: the measurement cannot separate them.
        app_module.predict_model_mask = lambda img, quality: shared.copy()
        try:
            quality, mask, graphic = choose_automatic_mask(image)
        finally:
            app_module.predict_model_mask = original

        self.assertEqual(quality, AUTO_PREFERRED_QUALITY)

    def test_a_model_that_cannot_run_is_skipped_not_fatal(self):
        image = self.edge_scene()
        import app as app_module

        original = app_module.predict_model_mask
        survivor = AUTO_PHOTO_CANDIDATES[-1]

        def one_model_fails(img, quality):
            if quality != survivor:
                raise MemoryError("no room for this model")
            return self.mask_from_box((60, 40, 180, 140))

        app_module.predict_model_mask = one_model_fails
        try:
            quality, mask, graphic = choose_automatic_mask(image)
        finally:
            app_module.predict_model_mask = original

        self.assertEqual(quality, survivor)

    def test_the_route_accepts_automatic_and_reports_its_choice(self):
        import app as app_module

        original = app_module.create_cutout
        app_module.create_cutout = lambda *a, **k: (cutout_bytes(), "balanced")
        try:
            response = app.test_client().post(
                "/remove",
                data={
                    "image": (io.BytesIO(cutout_bytes()), "sample.png"),
                    "quality": AUTO_QUALITY,
                },
                content_type="multipart/form-data",
            )
        finally:
            app_module.create_cutout = original

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Cutout-Quality"], "balanced")

    def test_tie_margin_separates_the_measured_cases(self):
        # Measured margins: 34% on the traffic photograph, 0.02% on hair.
        self.assertLess(AUTO_TIE_MARGIN, 0.34)
        self.assertGreater(AUTO_TIE_MARGIN, 0.0002)


if __name__ == "__main__":
    unittest.main()
