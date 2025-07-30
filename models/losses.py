import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module


class HorizonAwareLoss(Module):
    """
    Базовый класс для loss функций, которые учитывают горизонты прогноза.

    Основная идея: нормализуем ошибки относительно горизонта прогноза,
    чтобы сделать их сопоставимыми между разными горизонтами.

    Attributes
    ----------
    normalize_by_horizon : bool
        Включить ли нормализацию по горизонту
    weight_beta : float
        Параметр для вычисления весов по горизонту
    learnable_norm : LearnableHorizonNorm
        Обучаемый слой нормализации
    """

    def __init__(
        self, normalize_by_horizon: bool = True, norm_alpha_init: float = 0.3, weight_beta: float = 0.7
    ) -> None:
        super().__init__()
        self.normalize_by_horizon = normalize_by_horizon
        self.weight_beta = weight_beta

        from models.model import LearnableHorizonNorm

        self.learnable_norm = LearnableHorizonNorm(init_alpha=norm_alpha_init)

    def _normalize_by_horizon(self, loss: Tensor, horizon_hours: Tensor) -> Tensor:
        """
        Нормализует loss относительно горизонта прогноза.

        Parameters
        ----------
        loss : Tensor
            Исходный loss (например, расстояние в км)
        horizon_hours : Tensor
            Горизонты прогноза в часах

        Returns
        -------
        Tensor
            Нормализованный loss
        """
        if not self.normalize_by_horizon:
            return loss
        result: Tensor = self.learnable_norm(loss, horizon_hours)
        return result

    def _compute_horizon_weights(self, horizon_hours: Tensor) -> Tensor:
        """
        Вычисляет веса для разных горизонтов прогноза.

        Parameters
        ----------
        horizon_hours : Tensor
            Горизонты прогноза в часах

        Returns
        -------
        Tensor
            Веса для каждого сэмпла
        """
        weights = torch.pow(horizon_hours, self.weight_beta)
        result: Tensor = weights / weights.mean()
        return result


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
        radius_th: float = 150.0,  # Снижен с 300 до 150 км - более агрессивный порог
        smooth_km: float = 30.0,  # Снижен с 40 до 30 км - более резкий переход
        sector_weight: float = 0.8,  # Снижен с 1.0 до 0.8 - умеренный штраф за направление
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


class HorizonAwareSectorLoss(HorizonAwareLoss, HaversineDistanceMixin, DirectionalLossMixin):
    """
    Секторальный лосс с учетом горизонтов прогноза.

    Нормализует ошибки относительно горизонта прогноза для лучшего
    обучения на разных временных интервалах.

    Parameters
    ----------
    radius_th : float
        Порог расстояния (в км)
    smooth_km : float
        Ширина плавного перехода порога (в км)
    sector_weight : float
        Вес штрафа за направление
    earth_radius_km : float
        Радиус Земли в километрах
    normalize_by_horizon : bool
        Включить ли нормализацию по горизонту
    norm_alpha_init : float
        Начальное значение alpha для нормализации
    weight_beta : float
        Параметр для весов по горизонту
    radius_norm_coef : float
        Коэффициент нормализации радиуса
    """

    def __init__(
        self,
        radius_th: float = 150.0,  # Снижен с 300 до 150 км - более агрессивный порог
        smooth_km: float = 30.0,  # Снижен с 40 до 30 км - более резкий переход
        sector_weight: float = 0.8,  # Снижен с 1.0 до 0.8 - умеренный штраф за направление
        earth_radius_km: float = 6371.0,
        normalize_by_horizon: bool = True,
        norm_alpha_init: float = 0.25,  # Снижен с 0.3 до 0.25 - более сильная нормализация
        weight_beta: float = 0.6,  # Снижен с 0.7 до 0.6 - меньше веса длинным горизонтам
        radius_norm_coef: float = 20.0,  # Снижен с 24.0 до 20.0 - более агрессивная нормализация радиуса
    ):
        HorizonAwareLoss.__init__(self, normalize_by_horizon, norm_alpha_init, weight_beta)
        HaversineDistanceMixin.__init__(self, earth_radius_km)
        DirectionalLossMixin.__init__(self)

        self.radius_th = radius_th
        self.smooth_km = smooth_km
        self.sector_weight = sector_weight
        self.radius_norm_coef = radius_norm_coef

    def forward(
        self, preds: Tensor, target: Tensor, horizon_hours: Tensor, sample_weight: Tensor | None = None
    ) -> Tensor:
        """
        Вычисляет лосс с учетом горизонтов прогноза.

        Parameters
        ----------
        preds : Tensor
            Предсказанные изменения координат (dlat, dlon)
        target : Tensor
            Целевые изменения координат (dlat, dlon)
        horizon_hours : Tensor
            Горизонты прогноза в часах для каждого сэмпла
        sample_weight : Tensor | None, optional
            Веса семплов

        Returns
        -------
        Tensor
            Нормализованный лосс
        """
        # Вычисляем Haversine расстояние
        distance_km = self._compute_haversine_distance(preds, target)

        # Нормализуем расстояние относительно горизонта
        normalized_distance = self._normalize_by_horizon(distance_km, horizon_hours)

        # Базовая ошибка (нормализованное Haversine distance)
        base_loss = normalized_distance

        # Вес по радиусу (гладкая ступень) - используем настраиваемый коэффициент
        w_r = 1.0 / (
            1.0
            + torch.exp(
                -(normalized_distance - self.radius_th / self.radius_norm_coef)
                / (self.smooth_km / self.radius_norm_coef)
            )
        )

        # Штраф за сектор (направление)
        sector_pen = self._compute_directional_loss(preds, target, self.sector_weight)

        # Веса по горизонту
        horizon_weights = self._compute_horizon_weights(horizon_hours)

        # Итоговый вес
        weight = w_r * (1.0 + sector_pen) * horizon_weights
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


class HorizonAwareNLLGaussianLoss(HorizonAwareLoss):
    """
    NLL Gaussian loss с учетом горизонтов прогноза.

    Нормализует NLL относительно горизонта прогноза для лучшего обучения
    на разных временных интервалах.
    """

    def __init__(self, normalize_by_horizon: bool = True, norm_alpha_init: float = 0.3, weight_beta: float = 0.7):
        super().__init__(normalize_by_horizon, norm_alpha_init, weight_beta)

    def forward(
        self, preds: Tensor, target: Tensor, horizon_hours: Tensor, sample_weight: Tensor | None = None
    ) -> Tensor:
        """
        Вычисляет нормализованный NLL с учетом горизонта.

        Parameters
        ----------
        preds : Tensor
            Предсказания в формате (B, 4) -> mu_lat, mu_lon, log_var_lat, log_var_lon
        target : Tensor
            Целевые значения (B, 2) -> target_lat, target_lon
        horizon_hours : Tensor
            Горизонты прогноза в часах
        sample_weight : Tensor | None, optional
            Веса семплов

        Returns
        -------
        Tensor
            Нормализованный NLL
        """
        # preds: (B,4) -> mu_lat, mu_lon, log_var_lat, log_var_lon
        mu = preds[:, :2]
        log_var = preds[:, 2:]
        var = torch.exp(log_var).clamp(min=1e-6)
        nll = 0.5 * (log_var + (target - mu) ** 2 / var)
        nll = nll.sum(dim=1)  # sum over lat/lon

        # Нормализуем по горизонту
        normalized_nll = self._normalize_by_horizon(nll, horizon_hours)

        # Веса по горизонту
        horizon_weights = self._compute_horizon_weights(horizon_hours)

        if sample_weight is not None:
            horizon_weights = horizon_weights * sample_weight

        result: Tensor = (horizon_weights * normalized_nll).mean()
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


class HorizonAwareHaversineLoss(HorizonAwareLoss, HaversineDistanceMixin):
    """
    Haversine loss с учетом горизонтов прогноза.

    Нормализует расстояния относительно горизонта прогноза для лучшего
    обучения на разных временных интервалах.

    Parameters
    ----------
    earth_radius_km : float
        Радиус Земли в километрах
    max_speed_kmh : float
        Максимальная допустимая скорость в км/ч
    normalize_by_horizon : bool
        Включить ли нормализацию по горизонту
    norm_alpha_init : float
        Начальное значение alpha для нормализации
    weight_beta : float
        Параметр для весов по горизонту
    """

    def __init__(
        self,
        earth_radius_km: float = 6371.0,
        max_speed_kmh: float = 100.0,
        normalize_by_horizon: bool = True,
        norm_alpha_init: float = 0.3,
        weight_beta: float = 0.7,
    ):
        HorizonAwareLoss.__init__(self, normalize_by_horizon, norm_alpha_init, weight_beta)
        HaversineDistanceMixin.__init__(self, earth_radius_km)
        self.max_speed_kmh = max_speed_kmh

    def forward(
        self, preds: Tensor, target: Tensor, horizon_hours: Tensor, sample_weight: Tensor | None = None
    ) -> Tensor:
        """
        Вычисляет нормализованный Haversine loss.

        Parameters
        ----------
        preds : Tensor (B, 2)
            Предсказанные изменения координат (dlat, dlon) в градусах
        target : Tensor (B, 2)
            Целевые изменения координат (dlat, dlon) в градусах
        horizon_hours : Tensor
            Горизонты прогноза в часах для каждого сэмпла
        sample_weight : Tensor | None, optional
            Веса семплов

        Returns
        -------
        Tensor
            Нормализованный Haversine loss
        """
        # Вычисляем Haversine расстояние
        distance_km = self._compute_haversine_distance(preds, target)

        # Нормализуем расстояние относительно горизонта
        normalized_distance = self._normalize_by_horizon(distance_km, horizon_hours)

        # Веса по горизонту
        horizon_weights = self._compute_horizon_weights(horizon_hours)

        if sample_weight is not None:
            horizon_weights = horizon_weights * sample_weight

        return (horizon_weights * normalized_distance).mean()


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


class HorizonAwareCombinedCycloneLoss(HorizonAwareLoss, HaversineDistanceMixin, DirectionalLossMixin, SpeedLossMixin):
    """
    Комбинированный loss с учетом горизонтов прогноза.

    Нормализует все компоненты относительно горизонта прогноза для лучшего
    обучения на разных временных интервалах.

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
    normalize_by_horizon : bool
        Включить ли нормализацию по горизонту
    norm_alpha_init : float
        Начальное значение alpha для нормализации
    weight_beta : float
        Параметр для весов по горизонту
    """

    def __init__(
        self,
        haversine_weight: float = 1.0,
        direction_weight: float = 0.3,
        speed_weight: float = 0.1,
        max_speed_kmh: float = 100.0,
        earth_radius_km: float = 6371.0,
        normalize_by_horizon: bool = True,
        norm_alpha_init: float = 0.3,
        weight_beta: float = 0.7,
    ):
        HorizonAwareLoss.__init__(self, normalize_by_horizon, norm_alpha_init, weight_beta)
        HaversineDistanceMixin.__init__(self, earth_radius_km)
        DirectionalLossMixin.__init__(self)
        SpeedLossMixin.__init__(self, max_speed_kmh)

        self.haversine_weight = haversine_weight
        self.direction_weight = direction_weight
        self.speed_weight = speed_weight

        # Инициализируем HorizonAwareHaversineLoss один раз в __init__
        self.hav_loss = HorizonAwareHaversineLoss(
            earth_radius_km, max_speed_kmh, normalize_by_horizon, norm_alpha_init, weight_beta
        )

    def forward(
        self,
        preds: Tensor,
        target: Tensor,
        horizon_hours: Tensor,
        sample_weight: Tensor | None = None,
        current_lat: Tensor | None = None,
    ) -> Tensor:
        """
        Вычисляет комбинированный loss с учетом горизонтов.

        Parameters
        ----------
        preds : Tensor (B, 2)
            Предсказанные изменения координат (dlat, dlon)
        target : Tensor (B, 2)
            Целевые изменения координат (dlat, dlon)
        horizon_hours : Tensor
            Горизонты прогноза в часах для каждого сэмпла
        sample_weight : Tensor | None, optional
            Веса семплов
        current_lat : Tensor | None, optional
            Текущие широты для адаптивных весов

        Returns
        -------
        Tensor
            Нормализованный комбинированный loss
        """
        # 1. Haversine distance component (нормализованный)
        haversine_loss = self.hav_loss(preds, target, horizon_hours, sample_weight)

        # 2. Directional component (cosine similarity) - не нормализуем, так как это безразмерная величина
        pred_norm = torch.norm(preds, dim=1, keepdim=True)
        target_norm = torch.norm(target, dim=1, keepdim=True)

        # Избегаем деления на ноль
        pred_norm = torch.clamp(pred_norm, min=1e-8)
        target_norm = torch.clamp(target_norm, min=1e-8)

        pred_normalized = preds / pred_norm
        target_normalized = target / target_norm

        cosine_sim = torch.sum(pred_normalized * target_normalized, dim=1)
        direction_loss = (1.0 - cosine_sim).mean()

        # 3. Speed component (нормализованный)
        speed_penalty = self._compute_speed_loss(preds, horizon_hours, current_lat)

        # Нормализуем штраф скорости
        normalized_speed_penalty = self._normalize_by_horizon(speed_penalty, horizon_hours)
        speed_loss = normalized_speed_penalty.mean()

        # Комбинируем компоненты
        total_loss: Tensor = (
            self.haversine_weight * haversine_loss
            + self.direction_weight * direction_loss
            + self.speed_weight * speed_loss
        )

        return total_loss


class ImprovedSectorLoss(Module, HaversineDistanceMixin, DirectionalLossMixin):
    """
    Улучшенная Sector Loss с лучшей сходимостью.

    Особенности:
    1. Focal loss компонент для сложных случаев
    2. Адаптивные веса для разных горизонтов
    3. Smooth L1 loss вместо MSE для стабильности
    4. Динамическая нормализация

    Parameters
    ----------
    radius_th : float
        Порог расстояния (в км)
    smooth_km : float
        Ширина плавного перехода порога (в км)
    sector_weight : float
        Вес штрафа за направление
    earth_radius_km : float
        Радиус Земли в километрах
    focal_alpha : float
        Параметр alpha для focal loss
    focal_gamma : float
        Параметр gamma для focal loss
    use_smooth_l1 : bool
        Использовать ли Smooth L1 loss
    adaptive_weights : bool
        Использовать ли адаптивные веса
    """

    def __init__(
        self,
        radius_th: float = 150.0,
        smooth_km: float = 30.0,
        sector_weight: float = 0.8,
        earth_radius_km: float = 6371.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_smooth_l1: bool = True,
        adaptive_weights: bool = True,
    ):
        super().__init__()
        self.radius_th = radius_th
        self.smooth_km = smooth_km
        self.sector_weight = sector_weight
        self.earth_radius_km = earth_radius_km
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_smooth_l1 = use_smooth_l1
        self.adaptive_weights = adaptive_weights

        # Smooth L1 loss для стабильности
        if use_smooth_l1:
            self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")

        # Адаптивные веса для разных горизонтов
        if adaptive_weights:
            self.horizon_weights = nn.Parameter(torch.ones(1))

    def _compute_focal_component(self, distances: Tensor) -> Tensor:
        """
        Вычисляет focal loss компонент для сложных случаев.

        Focal loss помогает модели лучше фокусироваться на сложных примерах.

        Parameters
        ----------
        distances : Tensor
            Расстояния до целевых точек

        Returns
        -------
        Tensor
            Focal weights для каждого сэмпла
        """
        # Нормализуем расстояния
        normalized_distances = distances / self.radius_th

        # Вычисляем focal weights
        focal_weights = (1 - normalized_distances) ** self.focal_gamma

        # Применяем alpha weighting
        alpha_weights = self.focal_alpha * (normalized_distances > 0.5).float() + (1 - self.focal_alpha)

        result: Tensor = focal_weights * alpha_weights
        return result

    def forward(self, preds: Tensor, target: Tensor, sample_weight: Tensor | None = None) -> Tensor:
        """
        Улучшенный forward pass с focal loss и адаптивными весами.

        Parameters
        ----------
        preds : Tensor
            Предсказанные изменения координат (dlat, dlon)
        target : Tensor
            Целевые изменения координат (dlat, dlon)
        sample_weight : Tensor | None, optional
            Веса семплов

        Returns
        -------
        Tensor
            Улучшенный sector loss
        """
        # Вычисляем Haversine расстояния
        distances = self._compute_haversine_distance(preds, target)

        # Базовый loss с smooth L1 или MSE
        if self.use_smooth_l1:
            base_loss = self.smooth_l1_loss(preds, target).mean(dim=1)
        else:
            base_loss = torch.mean((preds - target) ** 2, dim=1)

        # Focal loss компонент
        focal_weights = self._compute_focal_component(distances)

        # Секторный loss
        sector_loss = self._compute_directional_loss(preds, target, self.sector_weight)

        # Комбинируем все компоненты
        combined_loss = base_loss * focal_weights + sector_loss

        # Применяем sample weights если предоставлены
        if sample_weight is not None:
            combined_loss = combined_loss * sample_weight

        result: Tensor = combined_loss.mean()
        return result


class ImprovedHorizonAwareSectorLoss(HorizonAwareLoss, HaversineDistanceMixin, DirectionalLossMixin):
    """
    Улучшенная Horizon-Aware Sector Loss с лучшей сходимостью.

    Особенности:
    1. Адаптивная нормализация по горизонтам
    2. Focal loss компонент
    3. Smooth L1 loss
    4. Динамические веса

    Parameters
    ----------
    radius_th : float
        Порог расстояния (в км)
    smooth_km : float
        Ширина плавного перехода порога (в км)
    sector_weight : float
        Вес штрафа за направление
    earth_radius_km : float
        Радиус Земли в километрах
    normalize_by_horizon : bool
        Включить ли нормализацию по горизонту
    norm_alpha_init : float
        Начальное значение alpha для нормализации
    weight_beta : float
        Параметр для весов по горизонту
    focal_alpha : float
        Параметр alpha для focal loss
    focal_gamma : float
        Параметр gamma для focal loss
    use_smooth_l1 : bool
        Использовать ли Smooth L1 loss
    """

    def __init__(
        self,
        radius_th: float = 150.0,
        smooth_km: float = 30.0,
        sector_weight: float = 0.8,
        earth_radius_km: float = 6371.0,
        normalize_by_horizon: bool = True,
        norm_alpha_init: float = 0.25,
        weight_beta: float = 0.6,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_smooth_l1: bool = True,
    ):
        super().__init__(
            normalize_by_horizon=normalize_by_horizon, norm_alpha_init=norm_alpha_init, weight_beta=weight_beta
        )
        self.radius_th = radius_th
        self.smooth_km = smooth_km
        self.sector_weight = sector_weight
        self.earth_radius_km = earth_radius_km
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_smooth_l1 = use_smooth_l1

        # Smooth L1 loss для стабильности
        if use_smooth_l1:
            self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")

    def _compute_focal_component(self, distances: Tensor, horizon_hours: Tensor) -> Tensor:
        """
        Вычисляет focal loss компонент с учетом горизонта.

        Parameters
        ----------
        distances : Tensor
            Расстояния до целевых точек
        horizon_hours : Tensor
            Горизонты прогноза в часах

        Returns
        -------
        Tensor
            Focal weights с учетом горизонта
        """
        # Нормализуем расстояния относительно горизонта
        normalized_distances = distances / (self.radius_th * (horizon_hours / 24.0))

        # Вычисляем focal weights
        focal_weights = (1 - torch.clamp(normalized_distances, 0, 1)) ** self.focal_gamma

        # Применяем alpha weighting
        alpha_weights = self.focal_alpha * (normalized_distances > 0.5).float() + (1 - self.focal_alpha)

        result: Tensor = focal_weights * alpha_weights
        return result

    def forward(
        self, preds: Tensor, target: Tensor, horizon_hours: Tensor, sample_weight: Tensor | None = None
    ) -> Tensor:
        """
        Улучшенный forward pass с учетом горизонта и focal loss.

        Parameters
        ----------
        preds : Tensor
            Предсказанные изменения координат (dlat, dlon)
        target : Tensor
            Целевые изменения координат (dlat, dlon)
        horizon_hours : Tensor
            Горизонты прогноза в часах для каждого сэмпла
        sample_weight : Tensor | None, optional
            Веса семплов

        Returns
        -------
        Tensor
            Улучшенный horizon-aware sector loss
        """
        # Вычисляем Haversine расстояния
        distances = self._compute_haversine_distance(preds, target)

        # Базовый loss с smooth L1 или MSE
        if self.use_smooth_l1:
            base_loss = self.smooth_l1_loss(preds, target).mean(dim=1)
        else:
            base_loss = torch.mean((preds - target) ** 2, dim=1)

        # Focal loss компонент с учетом горизонта
        focal_weights = self._compute_focal_component(distances, horizon_hours)

        # Секторный loss
        sector_loss = self._compute_directional_loss(preds, target, self.sector_weight)

        # Комбинируем все компоненты
        combined_loss = base_loss * focal_weights + sector_loss

        # Нормализуем по горизонту
        normalized_loss = self._normalize_by_horizon(combined_loss, horizon_hours)

        # Вычисляем веса для разных горизонтов
        horizon_weights = self._compute_horizon_weights(horizon_hours)

        # Применяем веса
        weighted_loss = normalized_loss * horizon_weights

        # Применяем sample weights если предоставлены
        if sample_weight is not None:
            weighted_loss = weighted_loss * sample_weight

        return weighted_loss.mean()
