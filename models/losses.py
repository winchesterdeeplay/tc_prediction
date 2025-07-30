import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module


class HaversineDistanceMixin:
    """
    Миксин для вычисления Haversine расстояния между точками.

    Attributes
    ----------
    earth_radius_km : float
        Радиус Земли в километрах
    """

    def __init__(self, earth_radius_km: float = 6371.0):
        self.earth_radius_km = earth_radius_km

    def _compute_haversine_distance(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Вычисляет Haversine расстояние между предсказанными и целевыми координатами.

        Parameters
        ----------
        preds : Tensor
            Предсказанные изменения координат (dlat, dlon) в градусах
        target : Tensor
            Целевые изменения координат (dlat, dlon) в градусах

        Returns
        -------
        Tensor
            Расстояние в километрах
        """
        # Конвертируем в радианы для Haversine вычислений
        pred_rad = torch.deg2rad(preds)
        target_rad = torch.deg2rad(target)

        # Вычисляем Haversine расстояние
        dlat = target_rad[:, 0] - pred_rad[:, 0]
        dlon = target_rad[:, 1] - pred_rad[:, 1]

        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(pred_rad[:, 0]) * torch.cos(target_rad[:, 0]) * torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))
        distance_km = self.earth_radius_km * c

        return distance_km


class DirectionalLossMixin:
    """Миксин для вычисления потерь, связанных с направлением."""

    def _compute_directional_loss(self, preds: Tensor, target: Tensor, sector_weight: float = 1.0) -> Tensor:
        """
        Вычисляет потерю, связанную с неправильным направлением.

        Parameters
        ----------
        preds : Tensor
            Предсказанные изменения координат (dlat, dlon)
        target : Tensor
            Целевые изменения координат (dlat, dlon)
        sector_weight : float
            Вес штрафа за направление

        Returns
        -------
        Tensor
            Потеря направления
        """
        dot = (preds * target).sum(dim=1)
        norm_prod = torch.norm(preds, dim=1) * torch.norm(target, dim=1) + 1e-6
        cos_theta = dot / norm_prod
        result: Tensor = sector_weight * 0.5 * (1.0 - cos_theta)
        return result


class SpeedLossMixin:
    """
    Миксин для вычисления потерь, связанных со скоростью.

    Attributes
    ----------
    max_speed_kmh : float
        Максимальная допустимая скорость в км/ч
    """

    def __init__(self, max_speed_kmh: float = 100.0):
        self.max_speed_kmh = max_speed_kmh

    def _compute_speed_loss(self, preds: Tensor, horizon_hours: Tensor, current_lat: Tensor | None = None) -> Tensor:
        """
        Вычисляет потерю за превышение максимальной скорости.

        Parameters
        ----------
        preds : Tensor
            Предсказанные изменения координат (dlat, dlon)
        horizon_hours : Tensor
            Горизонты прогноза в часах
        current_lat : Tensor | None, optional
            Текущие широты для точного вычисления расстояний в км

        Returns
        -------
        Tensor
            Штраф за скорость
        """
        if current_lat is not None:
            # Точное вычисление расстояния в км с учетом широты
            # Формула для км на градус: 111.13209 - 0.56605 * cos(2*lat) + 0.00120 * cos(4*lat)
            lat_rad = torch.deg2rad(current_lat)
            km_per_deg_lat = 111.13209 - 0.56605 * torch.cos(2 * lat_rad) + 0.00120 * torch.cos(4 * lat_rad)
            km_per_deg_lon = km_per_deg_lat * torch.cos(lat_rad)

            # Вычисляем расстояние в км
            dlat_km = preds[:, 0] * km_per_deg_lat
            dlon_km = preds[:, 1] * km_per_deg_lon
            distance_km = torch.sqrt(dlat_km**2 + dlon_km**2)
        else:
            deg2km = 111.0  # приблизительно
            distance_km = torch.norm(preds, dim=1) * deg2km

        # Конвертируем в скорость (км/ч)
        pred_speed_kmh = distance_km / horizon_hours

        # Штраф за превышение максимальной скорости
        return torch.relu(pred_speed_kmh - self.max_speed_kmh)


class SectorLoss(Module, HaversineDistanceMixin, DirectionalLossMixin):
    """
    Секторальный лосс для предсказания траекторий циклонов на основе Haversine расстояния.

    Лосс состоит из базовой ошибки Haversine и динамического веса, который:
    1. Увеличивается с ростом расстояния ошибки (в км)
    2. Учитывает направление вектора ошибки

    Parameters
    ----------
    radius_th : float
        Порог расстояния (в км), после которого ошибка считается "дальней"
    smooth_km : float
        Ширина плавного перехода порога (в км)
    sector_weight : float
        Максимальный дополнительный множитель за неправильный сектор
    earth_radius_km : float
        Радиус Земли в километрах
    """

    def __init__(
        self,
        radius_th: float = 200.0,
        smooth_km: float = 50.0,
        sector_weight: float = 1.0,
        earth_radius_km: float = 6371.0,
    ):
        Module.__init__(self)
        HaversineDistanceMixin.__init__(self, earth_radius_km)
        DirectionalLossMixin.__init__(self)

        self.radius_th = radius_th
        self.smooth_km = smooth_km
        self.sector_weight = sector_weight

    def forward(self, preds: Tensor, target: Tensor, sample_weight: Tensor | None = None) -> Tensor:
        """
        Вычисляет лосс для батча предсказаний и целевых значений.

        Parameters
        ----------
        preds : Tensor
            Предсказанные изменения координат (dlat, dlon)
        target : Tensor
            Целевые изменения координат (dlat, dlon)
        sample_weight : Tensor | None, optional
            Веса семплов для усреднения лосса

        Returns
        -------
        Tensor
            Среднее значение лосса по батчу
        """
        # Вычисляем Haversine расстояние
        distance_km = self._compute_haversine_distance(preds, target)

        # Базовая ошибка (Haversine distance в км)
        base_loss = distance_km

        # Вес по радиусу (гладкая ступень) - теперь в км
        w_r = 1.0 / (1.0 + torch.exp(-(distance_km - self.radius_th) / self.smooth_km))

        # Штраф за сектор (направление)
        sector_pen = self._compute_directional_loss(preds, target, self.sector_weight)

        # Итоговый вес
        weight = w_r * (1.0 + sector_pen)
        if sample_weight is not None:
            weight = weight * sample_weight

        result: Tensor = (weight * base_loss).mean()
        return result


class NLLGaussianLoss(Module):
    """
    Negative log-likelihood для 2D независимого Гауссова распределения с предсказанным log-variance.

    Эта loss функция позволяет модели предсказывать не только среднее значение,
    но и неопределенность предсказания.
    """

    def forward(self, preds: Tensor, target: Tensor, sample_weight: Tensor | None = None) -> Tensor:
        """
        Вычисляет NLL для Гауссова распределения.

        Parameters
        ----------
        preds : Tensor
            Предсказания в формате (B, 4) -> mu_lat, mu_lon, log_var_lat, log_var_lon
        target : Tensor
            Целевые значения (B, 2) -> target_lat, target_lon
        sample_weight : Tensor | None, optional
            Веса семплов

        Returns
        -------
        Tensor
            Средний NLL по батчу
        """
        # preds: (B,4) -> mu_lat, mu_lon, log_var_lat, log_var_lon
        mu = preds[:, :2]
        log_var = preds[:, 2:]
        var = torch.exp(log_var).clamp(min=1e-6)
        nll = 0.5 * (log_var + (target - mu) ** 2 / var)
        nll = nll.sum(dim=1)  # sum over lat/lon
        if sample_weight is not None:
            nll = nll * sample_weight
        result: Tensor = nll.mean()
        return result


class HaversineLoss(Module, HaversineDistanceMixin):
    """
    Loss на основе реального расстояния (Haversine) между предсказанной и истинной позициями.

    Учитывает сферическую геометрию Земли и физические ограничения движения циклонов.

    Parameters
    ----------
    earth_radius_km : float
        Радиус Земли в километрах
    max_speed_kmh : float
        Максимальная допустимая скорость в км/ч
    """

    def __init__(self, earth_radius_km: float = 6371.0, max_speed_kmh: float = 100.0):
        Module.__init__(self)
        HaversineDistanceMixin.__init__(self, earth_radius_km)
        self.max_speed_kmh = max_speed_kmh

    def forward(self, preds: Tensor, target: Tensor, sample_weight: Tensor | None = None) -> Tensor:
        """
        Вычисляет Haversine loss.

        Parameters
        ----------
        preds : Tensor (B, 2)
            Предсказанные изменения координат (dlat, dlon) в градусах
        target : Tensor (B, 2)
            Целевые изменения координат (dlat, dlon) в градусах
        sample_weight : Tensor | None, optional
            Веса семплов

        Returns
        -------
        Tensor
            Haversine loss в километрах
        """
        # Вычисляем Haversine расстояние
        distance_km = self._compute_haversine_distance(preds, target)

        if sample_weight is not None:
            distance_km = distance_km * sample_weight

        return distance_km.mean()


class CombinedCycloneLoss(Module, HaversineDistanceMixin, DirectionalLossMixin, SpeedLossMixin):
    """
    Комбинированный loss для циклонов, объединяющий несколько подходов.

    Сочетает:
    1. Haversine distance (физическая точность)
    2. Directional penalty (правильность направления)
    3. Speed penalty (реалистичность скорости)

    Parameters
    ----------
    haversine_weight : float
        Вес компонента Haversine distance
    direction_weight : float
        Вес компонента направления
    speed_weight : float
        Вес компонента скорости
    max_speed_kmh : float
        Максимальная допустимая скорость в км/ч
    earth_radius_km : float
        Радиус Земли в километрах
    """

    def __init__(
        self,
        haversine_weight: float = 1.0,
        direction_weight: float = 0.3,
        speed_weight: float = 0.1,
        max_speed_kmh: float = 100.0,
        earth_radius_km: float = 6371.0,
    ):
        Module.__init__(self)
        HaversineDistanceMixin.__init__(self, earth_radius_km)
        DirectionalLossMixin.__init__(self)
        SpeedLossMixin.__init__(self, max_speed_kmh)

        self.haversine_weight = haversine_weight
        self.direction_weight = direction_weight
        self.speed_weight = speed_weight

        # Инициализируем HaversineLoss один раз в __init__
        self.hav_loss = HaversineLoss(earth_radius_km, max_speed_kmh)

    def forward(
        self,
        preds: Tensor,
        target: Tensor,
        sample_weight: Tensor | None = None,
        current_lat: Tensor | None = None,
        horizon_hours: Tensor | None = None,
    ) -> Tensor:
        """
        Вычисляет комбинированный loss.

        Parameters
        ----------
        preds : Tensor (B, 2)
            Предсказанные изменения координат (dlat, dlon)
        target : Tensor (B, 2)
            Целевые изменения координат (dlat, dlon)
        sample_weight : Tensor | None, optional
            Веса семплов
        current_lat : Tensor | None, optional
            Текущие широты для адаптивных весов
        horizon_hours : Tensor | None, optional
            Горизонты прогноза в часах для проверки скорости

        Returns
        -------
        Tensor
            Комбинированный loss
        """
        # 1. Haversine distance component
        haversine_loss = self.hav_loss(preds, target, sample_weight)

        # 2. Directional component (cosine similarity)
        pred_norm = torch.norm(preds, dim=1, keepdim=True)
        target_norm = torch.norm(target, dim=1, keepdim=True)

        # Избегаем деления на ноль
        pred_norm = torch.clamp(pred_norm, min=1e-8)
        target_norm = torch.clamp(target_norm, min=1e-8)

        pred_normalized = preds / pred_norm
        target_normalized = target / target_norm

        cosine_sim = torch.sum(pred_normalized * target_normalized, dim=1)
        direction_loss = (1.0 - cosine_sim).mean()  # 0 = одинаковое направление, 2 = противоположное

        # 3. Speed component (если доступен горизонт)
        speed_loss = torch.zeros_like(direction_loss)
        if horizon_hours is not None:
            speed_penalty = self._compute_speed_loss(preds, horizon_hours, current_lat)
            speed_loss = speed_penalty.mean()

        # Комбинируем компоненты
        total_loss: Tensor = (
            self.haversine_weight * haversine_loss
            + self.direction_weight * direction_loss
            + self.speed_weight * speed_loss
        )

        return total_loss
