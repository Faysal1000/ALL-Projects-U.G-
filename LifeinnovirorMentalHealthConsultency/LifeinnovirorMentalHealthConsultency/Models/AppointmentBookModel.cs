using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Models
{
    public class AppointmentBookModel
    {
        [Required(ErrorMessage = "Full name is required.")]
        [StringLength(100)]
        public string FullName { get; set; }

        [Required(ErrorMessage = "Email is required.")]
        [EmailAddress(ErrorMessage = "Invalid email format.")]
        public string Email { get; set; }

        [Required(ErrorMessage = "Doctor ID is required.")]
        public int DoctorId { get; set; }

        [Required(ErrorMessage = "Slot ID is required.")]
        public int SlotId { get; set; }

        [Required(ErrorMessage = "Appointment Type ID is required.")]
        public int AppointmentTypeId { get; set; }

        [Required(ErrorMessage = "Meeting medium is required.")]
        [StringLength(20)]
        [RegularExpression("Online|Offline", ErrorMessage = "MeetingMedium must be Online or Offline")]
        public string MeetingMedium { get; set; }

        [StringLength(1000)]
        public string Notes { get; set; }
    }
}