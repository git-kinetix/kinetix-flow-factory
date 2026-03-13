# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/logger/clearml.py
from typing import Any, Dict, Optional
from .abc import Logger
from .formatting import LogImage, LogVideo, LogTable


class ClearMLLogger(Logger):
    def _init_platform(self):
        from clearml import Task as ClearMLTask, Logger as _ClearMLLogger

        task = ClearMLTask.init(
            project_name=self.config.project,
            task_name=self.config.run_name,
            auto_connect_frameworks=False,
        )
        task.connect(self.config.to_dict())
        self.platform = _ClearMLLogger.current_logger()
        self._task = task

    def _convert_to_platform(
        self,
        value: Any,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Any:
        """Convert to ClearML-compatible format (returns tuple of (type, data))."""
        if isinstance(value, LogImage):
            path = value.get_value(height, width)
            return ('image', path, value.caption)

        if isinstance(value, LogVideo):
            path = value.get_value(format='mp4', height=height, width=width)
            return ('video', path, value.caption)

        if isinstance(value, LogTable):
            h = height or value.target_height
            items = []
            for row in value.rows:
                for item in row:
                    if item is not None:
                        items.append(self._convert_to_platform(item, height=h))
            return ('table', items, value.columns)

        return ('scalar', value, None)

    def _log_impl(self, data: Dict, step: int):
        for key, value in data.items():
            self._log_single(key, value, step)
        self.platform.flush()

    def _log_single(self, key: str, value: Any, step: int):
        """Log a single converted value."""
        if isinstance(value, list):
            for i, v in enumerate(value):
                self._log_single(f"{key}/{i}", v, step)
            return

        if not isinstance(value, tuple):
            if isinstance(value, (int, float)):
                self.platform.report_scalar(title=key, series="value", value=value, iteration=step)
            return

        dtype, *args = value
        if dtype == 'scalar' and isinstance(args[0], (int, float)):
            self.platform.report_scalar(title=key, series="value", value=args[0], iteration=step)
        elif dtype == 'image':
            path, caption = args[0], args[1]
            series = caption or "image"
            self.platform.report_image(title=key, series=series, local_path=path, iteration=step)
        elif dtype == 'video':
            path, caption = args[0], args[1]
            series = caption or "video"
            self.platform.report_media(title=key, series=series, local_path=path, iteration=step)
        elif dtype == 'table':
            items, columns = args
            for i, item in enumerate(items):
                col_idx = i % len(columns)
                row_idx = i // len(columns)
                self._log_single(f"{key}/{columns[col_idx]}/{row_idx}", item, step)

    def __del__(self):
        if hasattr(self, '_task'):
            self._task.close()
